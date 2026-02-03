"""SIN3D++ Shutter Integration: Exposure integral along trajectory.

Priority A (correct smear): use_center_only=True (default) — at each t sample only
  warp(center, disp(t)); no temporal blend of prev/next (one disp(t) is wrong for
  other frames → misregistration → ghosting). Proper multi-frame would need
  motion-compensated interpolation (inverse flows per frame).

P0/P1: When use_center_only=False, frame_float from t_eff; RS row-offset in t_eff.
P4: use_zbuffer_no_weight_norm: B = Σ w_i * V * radiance, no / weight_sum.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from .trajectories import get_rs_time_offset_map


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear light. [..., 3] in [0, 1] -> linear."""
    threshold = 0.04045
    linear = torch.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    return linear


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert linear light to sRGB. [..., 3] in [0, 1] -> sRGB."""
    threshold = 0.0031308
    srgb = torch.where(
        linear <= threshold,
        linear * 12.92,
        1.055 * (linear ** (1.0 / 2.4)) - 0.055
    )
    return srgb


def integrate_shutter(
    sharp_seq: torch.Tensor,  # [T, 3, H, W] or [B, T, 3, H, W] sharp frames in [0, 1] or [-1, 1]
    traj: torch.Tensor,  # [B, N, H, W, 2] trajectory field
    visibility: torch.Tensor,  # [B, N, H, W] visibility weights
    params,
    use_linear_light: bool = True,
    use_zbuffer_no_weight_norm: bool = False,
    return_linear: bool = False,
    use_center_only: bool = True,  # Priority A: True = only warp(center, disp(t)) → proper smear; False = temporal blend (misregistration)
) -> torch.Tensor:
    """Integrate along trajectory. use_center_only=True: I_i = warp(center, disp(t_i)) → correct smear. False: blend prev/next then warp (ghosting)."""
    if sharp_seq.dim() == 4:
        sharp_seq = sharp_seq.unsqueeze(0)
        single_batch = True
    else:
        single_batch = False

    B, T, C, H, W = sharp_seq.shape
    N = traj.shape[1]
    device, dtype = traj.device, traj.dtype
    center_idx = T // 2

    if sharp_seq.min() < 0:
        sharp_seq = (sharp_seq + 1.0) / 2.0

    if use_linear_light:
        sharp_seq_linear = srgb_to_linear(sharp_seq)
    else:
        sharp_seq_linear = sharp_seq

    gain = getattr(params, "exposure_gain", 1.0)
    if gain != 1.0:
        sharp_seq_linear = sharp_seq_linear * gain

    time_grid_result = params.get_time_grid()
    if isinstance(time_grid_result, tuple):
        time_grid, weights = time_grid_result
    else:
        time_grid = time_grid_result
        weights = torch.ones_like(time_grid) / len(time_grid)
    weights = (weights / weights.sum()).to(device)

    blur_linear = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    center_linear = sharp_seq_linear[:, center_idx]  # for fallback when weight_sum ≈ 0

    yg, xg = torch.meshgrid(torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing="ij")

    if use_center_only:
        # Baseline: only center frame warped by disp(t). disp(t) is in reference (center) coords → correct smear.
        for i in range(N):
            disp = traj[:, i]
            xc = xg.unsqueeze(0).expand(B, -1, -1) + disp[..., 0]
            yc = yg.unsqueeze(0).expand(B, -1, -1) + disp[..., 1]
            grid = torch.stack([2.0 * (xc + 0.5) / W - 1.0, 2.0 * (yc + 0.5) / H - 1.0], dim=-1)
            warped = F.grid_sample(center_linear, grid, mode="bilinear", padding_mode="border", align_corners=False)
            vis = visibility[:, i : i + 1]
            w = weights[i].view(1, 1, 1, 1) * vis
            blur_linear = blur_linear + warped * w
            weight_sum = weight_sum + w
    else:
        # Legacy: temporal blend then warp (one disp wrong for other frames → ghosting).
        rs = params.rolling_shutter_strength
        rs_y0 = getattr(params, "rs_y0", 0.0)
        rs_map = get_rs_time_offset_map(H, W, rs, params.shutter_length, y0=rs_y0, device=device, dtype=dtype)
        row_offset = rs_map
        t_grid = time_grid.view(1, N, 1, 1).to(device)
        t_eff = t_grid + row_offset
        frame_float = (t_eff + 1.0) * (T - 1) / 2.0
        frame_float = frame_float.clamp(0.0, float(T - 1) - 1e-5)

        for i in range(N):
            disp = traj[:, i]
            ff = frame_float[0, i]
            k0 = torch.floor(ff).long().clamp(0, T - 2)
            k1 = k0 + 1
            alpha = (ff - k0.float()).clamp(0.0, 1.0)
            k0 = k0.expand(B, H, W)
            k1 = k1.expand(B, H, W)
            alpha = alpha.expand(B, H, W)
            idx0 = k0.unsqueeze(1).unsqueeze(2).expand(B, 1, C, H, W).long()
            idx1 = k1.unsqueeze(1).unsqueeze(2).expand(B, 1, C, H, W).long()
            s0 = torch.gather(sharp_seq_linear, 1, idx0).squeeze(1)
            s1 = torch.gather(sharp_seq_linear, 1, idx1).squeeze(1)
            alpha_bc = alpha.unsqueeze(1).expand(B, C, H, W)
            I_i = (1.0 - alpha_bc) * s0 + alpha_bc * s1
            xc = xg.unsqueeze(0).expand(B, -1, -1) + disp[..., 0]
            yc = yg.unsqueeze(0).expand(B, -1, -1) + disp[..., 1]
            grid = torch.stack([2.0 * (xc + 0.5) / W - 1.0, 2.0 * (yc + 0.5) / H - 1.0], dim=-1)
            warped = F.grid_sample(I_i, grid, mode="bilinear", padding_mode="border", align_corners=False)
            vis = visibility[:, i : i + 1]
            w = weights[i].view(1, 1, 1, 1) * vis
            blur_linear = blur_linear + warped * w
            weight_sum = weight_sum + w

    if not use_zbuffer_no_weight_norm:
        weight_sum = weight_sum.clamp(min=1e-6)
        blur_linear = blur_linear / weight_sum
    else:
        # When z-buffer: avoid black pixels where weight_sum ≈ 0 (e.g. visibility≈0 at depth edges). Fallback to center.
        blend = (weight_sum > 1e-4).float()
        blur_linear = blur_linear * blend + center_linear * (1.0 - blend)

    if use_linear_light and return_linear:
        blur = blur_linear
    elif use_linear_light:
        blur = linear_to_srgb(blur_linear)
    else:
        blur = blur_linear

    if single_batch:
        blur = blur.squeeze(0)
    return blur


def warp(
    img: torch.Tensor,
    disp: torch.Tensor,
) -> torch.Tensor:
    """Warp image by displacement field. [B,3,H,W] / [3,H,W], [B,H,W,2] / [H,W,2] -> warped."""
    single_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        single_batch = True
    if disp.dim() == 3:
        disp = disp.unsqueeze(0)
    B, C, H, W = img.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij'
    )
    x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
    x_warped = x_coords + disp[..., 0]
    y_warped = y_coords + disp[..., 1]
    grid_x = 2.0 * (x_warped + 0.5) / W - 1.0
    grid_y = 2.0 * (y_warped + 0.5) / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)
    if single_batch:
        warped = warped.squeeze(0)
    return warped
