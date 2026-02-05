"""Trajectory field construction: camera and object motion."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def get_rs_time_offset(H: int, W: int, rs_factor: float, shutter_length: float, y0: float = 0.0,
                       device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Rolling shutter time offset: dt(y) = rs_factor * shutter_length * (y - y0) / H."""
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    return rs_factor * shutter_length * (y - y0 * H) / max(H, 1)


def _perpendicular(flow: torch.Tensor) -> torch.Tensor:
    """Perpendicular to 2D flow: (x, y) -> (-y, x)."""
    out = flow.clone()
    out[..., 0] = -flow[..., 1]
    out[..., 1] = flow[..., 0]
    return out


def _smooth_walk_path(N: int, jerk: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Integrated noise path for smooth_walk profile."""
    if jerk <= 0 or N < 2:
        return torch.linspace(-1, 1, N, device=device, dtype=dtype)
    dt = 2.0 / max(N - 1, 1)
    j = jerk * (2 * torch.rand(N, device=device, dtype=dtype) - 1)
    a = torch.zeros(N, device=device, dtype=dtype)
    for i in range(1, N):
        a[i] = (a[i-1] + j[i] * dt).clamp(-1, 1)
    v = torch.zeros(N, device=device, dtype=dtype)
    for i in range(1, N):
        v[i] = v[i-1] + a[i] * dt
    p = torch.cumsum(v * dt, dim=0)
    p = p - p[N // 2]
    return (p / p.abs().max().clamp(min=1e-6)).to(dtype)


def _rotate_vectors(vectors: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate 2D vectors by angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=vectors.device, dtype=vectors.dtype)
    
    if vectors.dim() == 4:  # [B, H, W, 2]
        return torch.einsum("bhwc,cd->bhwd", vectors, rot)
    elif vectors.dim() == 3:  # [H, W, 2]
        return torch.einsum("hwc,cd->hwd", vectors, rot)
    shape = vectors.shape
    return torch.matmul(vectors.view(-1, 2), rot).view(*shape)


def camera_motion(
    depth: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    params,
    num_subframes: int,
    camera_flow_fwd: Optional[torch.Tensor] = None,
    camera_flow_bwd: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build camera motion trajectory [B, N, H, W, 2]."""
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    
    if camera_flow_fwd is not None and camera_flow_bwd is not None:
        base_flow = 0.5 * (camera_flow_fwd - camera_flow_bwd).permute(0, 2, 3, 1)
    else:
        base_flow = 0.5 * (flow_fwd - flow_bwd).permute(0, 2, 3, 1)
    
    cam_rot = getattr(params, "camera_rotation_scale", 0.0)
    if abs(cam_rot) > 1e-6:
        base_flow = _rotate_vectors(base_flow, np.rad2deg(float(cam_rot)))
    
    from_pose = getattr(params, "_camera_from_pose", False)
    z_scale = (1 / (depth + 1e-3)).clamp(max=5).permute(0, 2, 3, 1) if from_pose else torch.ones(B, H, W, 1, device=device, dtype=dtype)
    
    scaled_flow = base_flow * params.camera_translation_scale * params.depth_parallax_scale * z_scale
    
    time_result = params.get_time_grid()
    time_grid = time_result[0] if isinstance(time_result, tuple) else time_result
    
    rs_map = get_rs_time_offset(H, W, params.rolling_shutter_strength, params.shutter_length, 
                                 getattr(params, "rs_y0", 0.0), device, dtype)
    row_offset = rs_map.view(1, H, 1)
    
    profile = getattr(params, "trajectory_profile", "constant")
    smooth_path = _smooth_walk_path(num_subframes, getattr(params, "smooth_walk_jerk", 0.0), device, dtype) if profile == "smooth_walk" else None
    
    traj = []
    for i, t in enumerate(time_grid):
        t_scaled = t * params.shutter_length
        if profile == "constant" and params.camera_jerk > 0:
            t_scaled = t_scaled * (1 + params.camera_jerk * t_scaled ** 2)
        t_eff = (t_scaled + row_offset).expand(B, H, W)
        
        if profile == "constant":
            factor = t_eff
        elif profile == "acceleration":
            a = getattr(params, "camera_acceleration", 0.0)
            factor = t_eff + 0.5 * a * t_eff ** 2
        elif profile == "smooth_walk" and smooth_path is not None:
            factor = smooth_path[i].view(1, 1, 1).expand(B, H, W)
        else:
            factor = t_eff
        
        disp = scaled_flow * factor.unsqueeze(-1)
        lateral = getattr(params, "lateral_acceleration", 0.0)
        if lateral != 0:
            disp = disp + lateral * 0.5 * t_eff.unsqueeze(-1) ** 2 * _perpendicular(scaled_flow)
        traj.append(disp)
    
    return torch.stack(traj, dim=1)


def object_motion(
    masks: Optional[torch.Tensor],
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    params,
    num_subframes: int,
) -> torch.Tensor:
    """Build object motion trajectory [B, N, H, W, 2]."""
    B, _, H, W = flow_fwd.shape
    device, dtype = flow_fwd.device, flow_fwd.dtype
    
    if masks is None or masks.shape[1] == 0 or masks.sum() < 100:
        return torch.zeros(B, num_subframes, H, W, 2, device=device, dtype=dtype)
    
    K = masks.shape[1]
    
    camera_flow_fwd = getattr(params, "_camera_flow_fwd", None)
    camera_flow_bwd = getattr(params, "_camera_flow_bwd", None)
    if camera_flow_fwd is not None and camera_flow_bwd is not None:
        res_fwd = flow_fwd - camera_flow_fwd
        res_bwd = flow_bwd - camera_flow_bwd
        base_flow = 0.5 * (res_fwd - res_bwd).permute(0, 2, 3, 1)
    else:
        res_fwd = flow_fwd
        res_bwd = flow_bwd
        base_flow = 0.5 * (flow_fwd - flow_bwd).permute(0, 2, 3, 1)
    
    yg, xg = torch.meshgrid(torch.arange(H, device=device, dtype=dtype), 
                            torch.arange(W, device=device, dtype=dtype), indexing="ij")
    
    time_result = params.get_time_grid()
    time_grid = time_result[0] if isinstance(time_result, tuple) else time_result
    
    rs_map = get_rs_time_offset(H, W, params.rolling_shutter_strength, params.shutter_length,
                                 getattr(params, "rs_y0", 0.0), device, dtype)
    row_dt = rs_map.view(1, H, 1)
    
    profile = getattr(params, "trajectory_profile", "constant")
    smooth_path = _smooth_walk_path(num_subframes, getattr(params, "smooth_walk_jerk", 0.0), device, dtype) if profile == "smooth_walk" else None
    
    # Scale and optionally rotate the real object motion (residual flow)
    # object_scale > 1 = amplify object motion, < 1 = dampen
    scaled_flow = base_flow * params.object_scale
    if params.object_direction != 0.0:
        scaled_flow = _rotate_vectors(scaled_flow, params.object_direction)
    
    traj = torch.zeros(B, num_subframes, H, W, 2, device=device, dtype=dtype)
    object_model = getattr(params, "object_model", "dense")
    
    # Object trajectory follows the SAME profile as camera (constant/acceleration)
    # but uses the scaled residual flow direction - no separate acceleration!
    # This prevents ghosting by keeping motion physically consistent
    cam_accel = getattr(params, "camera_acceleration", 0.0)
    
    def _dense_disp(i, t):
        t_scaled = t * params.shutter_length
        t_eff = (t_scaled + row_dt).expand(B, H, W)
        if profile == "constant":
            factor = t_eff
        elif profile == "acceleration":
            # Use camera acceleration for consistent trajectory shape
            factor = t_eff + 0.5 * cam_accel * t_eff ** 2
        elif profile == "smooth_walk" and smooth_path is not None:
            factor = smooth_path[i].view(1, 1, 1).expand(B, H, W)
        else:
            factor = t_eff
        disp = scaled_flow * factor.unsqueeze(-1)
        return disp
    
    for k in range(K):
        mask = masks[:, k:k+1]
        mask_exp = mask.permute(0, 2, 3, 1)
        
        if object_model == "se2":
            for b in range(B):
                mask_bk = masks[b, k]
                idx = (mask_bk > 0.5).flatten().nonzero().squeeze(-1)
                if idx.numel() < 10:
                    for i, t in enumerate(time_grid):
                        disp = _dense_disp(i, t)
                        traj[b, i] += disp[b] * mask_bk.unsqueeze(-1)
                    continue
                
                p = torch.stack([xg.flatten()[idx], yg.flatten()[idx]], dim=1)
                res_fwd_flat = res_fwd[b].permute(1, 2, 0).reshape(-1, 2)[idx]
                res_bwd_flat = res_bwd[b].permute(1, 2, 0).reshape(-1, 2)[idx]
                w = mask_bk.flatten()[idx]
                
                T_plus = _fit_se2(p, p + res_fwd_flat, w)
                T_minus = _fit_se2(p, p + res_bwd_flat, w)
                
                if T_plus is None or T_minus is None:
                    for i, t in enumerate(time_grid):
                        disp = _dense_disp(i, t)
                        traj[b, i] += disp[b] * mask_bk.unsqueeze(-1)
                    continue
                
                theta_p, v_p = _se2_log(T_plus.unsqueeze(0))
                T_minus_inv = _se2_inverse(T_minus.unsqueeze(0))
                theta_m, v_m = _se2_log(T_minus_inv)
                
                theta_sym = 0.5 * (theta_p + theta_m)
                v_sym = 0.5 * (v_p + v_m) * params.object_scale
                
                if params.object_direction != 0.0:
                    v_sym = _rotate_vectors(v_sym.unsqueeze(0), params.object_direction).squeeze(0)
                
                for i, t in enumerate(time_grid):
                    t_scaled = t * params.shutter_length
                    t_eff = (t_scaled + row_dt).expand(1, H, W)
                    if profile == "constant":
                        s_map = t_eff
                    elif profile == "acceleration":
                        # Use camera acceleration for consistent trajectory shape
                        s_map = t_eff + 0.5 * cam_accel * t_eff ** 2
                    elif profile == "smooth_walk" and smooth_path is not None:
                        s_map = smooth_path[i].view(1, 1, 1).expand(1, H, W)
                    else:
                        s_map = t_eff
                    disp = _se2_disp(theta_sym, v_sym, s_map, xg, yg)
                    traj[b, i] += disp[0] * mask_bk.unsqueeze(-1)
        else:
            for i, t in enumerate(time_grid):
                disp = _dense_disp(i, t) * mask_exp
                traj[:, i] += disp
    
    return traj


def build_trajectory(cam_traj: torch.Tensor, obj_traj: torch.Tensor) -> torch.Tensor:
    """Combine camera and object trajectories."""
    return cam_traj + obj_traj


# SE(2) utilities
def _fit_se2(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """Weighted 2D Procrustes for SE(2)."""
    if p.shape[0] < 10:
        return None
    w = w.clamp(min=0).reshape(-1, 1)
    wsum = w.sum().clamp(min=1e-10)
    p_bar = (w * p).sum(0) / wsum
    q_bar = (w * q).sum(0) / wsum
    p_, q_ = p - p_bar, q - q_bar
    a = (w.squeeze() * (p_[:, 0] * q_[:, 0] + p_[:, 1] * q_[:, 1])).sum()
    b = (w.squeeze() * (p_[:, 0] * q_[:, 1] - p_[:, 1] * q_[:, 0])).sum()
    theta = torch.atan2(b, a + 1e-10)
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)
    t = q_bar - R @ p_bar
    T = torch.eye(3, device=p.device, dtype=p.dtype)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def _se2_inverse(T: torch.Tensor) -> torch.Tensor:
    R, t = T[:, :2, :2], T[:, :2, 2:3]
    Rt = R.transpose(1, 2)
    Tout = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    Tout[:, :2, :2] = Rt
    Tout[:, :2, 2:3] = -Rt @ t
    return Tout


def _se2_log(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    R, t = T[:, :2, :2], T[:, :2, 2]
    theta = torch.atan2(R[:, 1, 0], R[:, 0, 0]).view(-1, 1, 1)
    A = _sinx_over_x(theta)
    B = _one_minus_cos_over_x(theta)
    det = (A * A + B * B).view(-1, 1)
    inv00, inv01 = A.view(-1, 1) / det, B.view(-1, 1) / det
    vx = inv00[:, 0] * t[:, 0] + inv01[:, 0] * t[:, 1]
    vy = -inv01[:, 0] * t[:, 0] + inv00[:, 0] * t[:, 1]
    return theta, torch.stack([vx, vy], dim=1)


def _se2_disp(theta: torch.Tensor, v: torch.Tensor, s: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor) -> torch.Tensor:
    B, H, W = s.shape
    alpha = s * theta.view(B, 1, 1)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    A, Bm = _sinx_over_x(alpha), _one_minus_cos_over_x(alpha)
    vx, vy = v[:, 0].view(B, 1, 1), v[:, 1].view(B, 1, 1)
    tx = A * vx - Bm * vy
    ty = Bm * vx + A * vy
    x = xg.view(1, H, W).expand(B, -1, -1)
    y = yg.view(1, H, W).expand(B, -1, -1)
    return torch.stack([ca * x - sa * y + tx - x, sa * x + ca * y + ty - y], dim=-1)


def _sinx_over_x(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x2 = x * x
    return torch.where(x.abs() < eps, 1 - x2/6 + x2*x2/120, torch.sin(x) / x)


def _one_minus_cos_over_x(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x2 = x * x
    return torch.where(x.abs() < eps, x/2 - x*x2/24 + x*x2*x2/720, (1 - torch.cos(x)) / x)
