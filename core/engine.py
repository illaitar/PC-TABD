"""BlurEngine: Main synthesis pipeline."""

import torch
from typing import Optional, Dict, Tuple, Union
import numpy as np

from .config import BlurConfig, BlurParams
from .trajectories import camera_motion, object_motion, build_trajectory, get_rs_time_offset
from .visibility import compute_visibility
from .shutter import integrate_shutter
from .isp import apply_isp
from .camera import extract_homography, extract_se2, homography_to_flow, extract_object_masks, compute_object_confidence, fit_se2_from_flow, se2_to_flow


class BlurEngine:
    """Blur synthesis engine."""
    
    def __init__(
        self,
        cfg: BlurConfig,
        extract_objects: bool = True,
        use_linear_light: bool = True,
        use_depth_ordering: bool = True,
    ):
        self.cfg = cfg
        self.residual_threshold = getattr(cfg, "object_residual_threshold", 2.0)
        self.extract_objects = extract_objects
        self.use_linear_light = use_linear_light
        self.use_depth_ordering = use_depth_ordering
    
    @torch.no_grad()
    def synthesize(
        self,
        sharp_seq: torch.Tensor,
        depth: torch.Tensor,
        flow_fwd: torch.Tensor,
        flow_bwd: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        occ_mask: Optional[torch.Tensor] = None,
        params: Optional[BlurParams] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Synthesize blur from sharp sequence.
        
        Args:
            sharp_seq: [T, 3, H, W] or [B, T, 3, H, W] sharp frames
            depth: [1, H, W] or [B, 1, H, W] depth map
            flow_fwd: [2, H, W] or [B, 2, H, W] forward flow
            flow_bwd: [2, H, W] or [B, 2, H, W] backward flow
            masks: optional [K, H, W] or [B, K, H, W] object masks
            occ_mask: optional [1, H, W] or [B, 1, H, W] occlusion mask
            params: optional BlurParams (samples from config if None)
        
        Returns:
            blur: [3, H, W] or [B, 3, H, W] blurred image
            meta: dict with trajectories, visibility, params
        """
        if params is None:
            params = self.cfg.sample()
        
        device = depth.device if isinstance(depth, torch.Tensor) else torch.device(self.cfg.device)
        params.device = str(device)
        
        single_batch = sharp_seq.dim() == 4
        if single_batch:
            sharp_seq = sharp_seq.unsqueeze(0)
        
        B, T, C, H, W = sharp_seq.shape
        center_idx = T // 2
        
        sharp_prev = sharp_seq[:, max(0, center_idx - 1)]
        sharp = sharp_seq[:, center_idx]
        sharp_next = sharp_seq[:, min(T - 1, center_idx + 1)]
        
        if flow_fwd.dim() == 3:
            flow_fwd = flow_fwd.unsqueeze(0)
        if flow_bwd.dim() == 3:
            flow_bwd = flow_bwd.unsqueeze(0)
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        
        # Extract camera motion
        camera_model = getattr(self.cfg, "camera_model", "homography")
        
        if camera_model == "flow":
            cam_fwd, cam_bwd = flow_fwd, flow_bwd
            res_fwd = torch.zeros_like(flow_fwd)
            res_bwd = torch.zeros_like(flow_bwd)
            if masks is None:
                masks = torch.zeros(B, 0, H, W, device=device, dtype=flow_fwd.dtype)
            params._cam_traj_se2 = None
        elif camera_model == "se2_rigid":
            # Fit camera SE2 from optical flow (RANSAC rejects moving objects)
            cam_se2_fwd = fit_se2_from_flow(flow_fwd[0] if flow_fwd.dim() == 4 else flow_fwd)
            cam_se2_bwd = fit_se2_from_flow(flow_bwd[0] if flow_bwd.dim() == 4 else flow_bwd)
            
            if cam_se2_fwd is None:
                cam_se2_fwd = torch.eye(3, device=device)
            if cam_se2_bwd is None:
                cam_se2_bwd = torch.eye(3, device=device)
            
            # Convert SE2 to flow
            se2_cam_fwd = se2_to_flow(cam_se2_fwd, H, W).unsqueeze(0)
            se2_cam_bwd = se2_to_flow(cam_se2_bwd, H, W).unsqueeze(0)
            
            # Residual = optical_flow - SE2_camera_flow
            res_fwd = flow_fwd - se2_cam_fwd
            res_bwd = flow_bwd - se2_cam_bwd
            
            # Compute soft object confidence
            saturation = getattr(self.cfg, "object_saturation", 10.0)
            obj_confidence = compute_object_confidence(res_fwd, res_bwd, self.residual_threshold, saturation)
            params._obj_confidence = obj_confidence
            
            # Extract binary masks for object tracking
            if self.extract_objects and masks is None:
                adaptive = getattr(self.cfg, "object_adaptive_threshold", False)
                masks = extract_object_masks(res_fwd, res_bwd, self.residual_threshold, adaptive=adaptive)
            
            # Fit object SE2 from object regions (where confidence > 0.5)
            obj_mask = (obj_confidence[0, 0] > 0.3) if obj_confidence.shape[0] > 0 else None
            obj_se2 = fit_se2_from_flow(flow_fwd[0] if flow_fwd.dim() == 4 else flow_fwd, obj_mask)
            
            if obj_se2 is not None:
                obj_flow = se2_to_flow(obj_se2, H, W).unsqueeze(0)
                obj_offset = obj_flow - se2_cam_fwd  # difference from camera
            else:
                obj_offset = torch.zeros_like(se2_cam_fwd)
            
            # Scale object offset by object_scale parameter
            object_scale = params.object_scale
            
            # Combined flow: camera + scaled object offset where objects are detected
            combined_fwd = se2_cam_fwd + obj_confidence * obj_offset * object_scale
            combined_bwd = se2_cam_bwd + obj_confidence * (-obj_offset) * object_scale  # reverse for backward
            
            # Store for trajectory building
            params._obj_offset = obj_offset
            params._object_scale = object_scale
            
            cam_fwd = combined_fwd
            cam_bwd = combined_bwd
            
            # Build camera trajectory using the old method (for consistency with other parts)
            T_prev = extract_se2(sharp, sharp_prev)
            T_next = extract_se2(sharp, sharp_next)
            cam_traj_se2, _, _ = self._camera_motion_se2(
                params, params.num_subframes, T_prev.unsqueeze(0) if T_prev.dim() == 2 else T_prev,
                T_next.unsqueeze(0) if T_next.dim() == 2 else T_next, H, W
            )
            params._cam_traj_se2 = cam_traj_se2
        else:
            H_prev, _ = extract_homography(sharp, sharp_prev)
            H_next, _ = extract_homography(sharp, sharp_next)
            cam_bwd = homography_to_flow(H_prev, H, W)
            cam_fwd = homography_to_flow(H_next, H, W)
            if cam_fwd.dim() == 3:
                cam_fwd, cam_bwd = cam_fwd.unsqueeze(0), cam_bwd.unsqueeze(0)
            res_fwd = flow_fwd - cam_fwd
            res_bwd = flow_bwd - cam_bwd
            if self.extract_objects and masks is None:
                adaptive = getattr(self.cfg, "object_adaptive_threshold", False)
                masks = extract_object_masks(res_fwd, res_bwd, self.residual_threshold, adaptive=adaptive)
            saturation = getattr(self.cfg, "object_saturation", 10.0)
            obj_confidence = compute_object_confidence(res_fwd, res_bwd, self.residual_threshold, saturation)
            params._obj_confidence = obj_confidence
            params._cam_traj_se2 = None
        
        params._camera_flow_fwd = cam_fwd
        params._camera_flow_bwd = cam_bwd
        params._camera_from_pose = False
        params.rs_y0 = getattr(self.cfg, "rs_y0", 0.0)
        params.object_residual_max_norm = getattr(self.cfg, "object_residual_max_norm", 2.0)
        params.visibility_floor = getattr(self.cfg, "visibility_floor", 0.12)
        
        # Build trajectories
        if getattr(params, "_cam_traj_se2", None) is not None:
            cam_traj = params._cam_traj_se2
        else:
            cam_traj = camera_motion(depth, flow_fwd, flow_bwd, params, params.num_subframes, cam_fwd, cam_bwd)
        
        obj_traj = object_motion(masks, flow_fwd, flow_bwd, params, params.num_subframes)
        traj = build_trajectory(cam_traj, obj_traj)
        
        # Visibility
        visibility = compute_visibility(traj, depth, params, occ_mask, self.use_depth_ordering)
        
        # Normalize sharp sequence
        if sharp_seq.min() < 0:
            sharp_seq = (sharp_seq + 1) / 2
        
        # Shutter integration
        has_objects = masks is not None and masks.shape[1] > 0
        obj_confidence = getattr(params, "_obj_confidence", None)
        soft_blend = getattr(self.cfg, "object_soft_blend", True)
        blur = integrate_shutter(
            sharp_seq, traj, visibility, params,
            use_linear_light=self.use_linear_light,
            use_zbuffer=self.use_depth_ordering,
            return_linear=self.use_linear_light,
            cam_traj=cam_traj if has_objects else None,
            obj_traj=obj_traj if has_objects else None,
            object_masks=masks if has_objects else None,
            object_mask_feather=getattr(self.cfg, "object_mask_feather", 3),
            object_confidence=obj_confidence if has_objects and soft_blend else None,
            bg_fill=getattr(self.cfg, "bg_fill", "inpaint"),
        )
        
        if blur.dim() == 3:
            blur = blur.unsqueeze(0)
        
        # ISP
        blur = apply_isp(blur, params, use_linear_input=self.use_linear_light)
        blur = blur.clamp(0, 1)
        
        if single_batch:
            blur = blur.squeeze(0)
        
        meta = {
            "traj": traj,
            "visibility": visibility,
            "params": params,
            "cam_traj": cam_traj,
            "obj_traj": obj_traj,
            "cam_fwd": cam_fwd,
            "cam_bwd": cam_bwd,
            "res_fwd": res_fwd,
            "res_bwd": res_bwd,
            "masks": masks,
        }
        
        return blur, meta
    
    def _se2_to_flow(self, T, H, W, device):
        """Convert SE2 transform to dense flow field."""
        # Handle batch dimension
        if T.dim() == 2:
            T = T.unsqueeze(0)
        B = T.shape[0]
        
        theta = torch.atan2(T[:, 1, 0], T[:, 0, 0])  # [B]
        tx, ty = T[:, 0, 2], T[:, 1, 2]  # [B], [B]
        
        yg, xg = torch.meshgrid(
            torch.arange(H, device=device, dtype=T.dtype),
            torch.arange(W, device=device, dtype=T.dtype),
            indexing="ij"
        )
        xg = xg.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        yg = yg.unsqueeze(0).expand(B, -1, -1)
        
        ca = torch.cos(theta).view(B, 1, 1)
        sa = torch.sin(theta).view(B, 1, 1)
        tx = tx.view(B, 1, 1)
        ty = ty.view(B, 1, 1)
        
        dx = ca * xg - sa * yg + tx - xg
        dy = sa * xg + ca * yg + ty - yg
        
        flow = torch.stack([dx, dy], dim=1)  # [B, 2, H, W]
        return flow.squeeze(0) if B == 1 else flow
    
    def _camera_motion_se2(self, params, num_subframes, T_prev, T_next, H, W):
        """Build SE(2) camera trajectory."""
        device, dtype = T_next.device, T_next.dtype
        B = T_next.shape[0]
        
        time_result = params.get_time_grid()
        time_grid = time_result[0] if isinstance(time_result, tuple) else time_result
        
        rs_map = get_rs_time_offset(H, W, params.rolling_shutter_strength, params.shutter_length,
                                     getattr(params, "rs_y0", 0.0), device, dtype)
        row_offset = rs_map.view(1, H, 1)
        
        yg, xg = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                torch.arange(W, device=device, dtype=dtype), indexing="ij")
        
        rot_scale = getattr(params, "se2_rotation_scale", 1.0)
        mode = getattr(self.cfg, "se2_mode", "symmetric")
        
        if mode == "symmetric":
            T_prev_inv = self._se2_inverse(T_prev)
            th_n, v_n = self._se2_log(T_next)
            th_p, v_p = self._se2_log(T_prev_inv)
            theta = 0.5 * (th_n + th_p) * rot_scale
            v = 0.5 * (v_n + v_p)
            
            s_plus = torch.ones(B, H, W, device=device, dtype=dtype)
            s_minus = -torch.ones(B, H, W, device=device, dtype=dtype)
            cam_fwd = self._se2_disp(theta, v, s_plus, xg, yg).permute(0, 3, 1, 2)
            cam_bwd = self._se2_disp(theta, v, s_minus, xg, yg).permute(0, 3, 1, 2)
            
            def disp_from_s(s_map):
                return self._se2_disp(theta, v, s_map, xg, yg)
        else:
            th_n, v_n = self._se2_log(T_next)
            th_p, v_p = self._se2_log(T_prev)
            th_n, th_p = th_n * rot_scale, th_p * rot_scale
            
            cam_fwd = self._se2_disp(th_n, v_n, torch.ones(B, H, W, device=device, dtype=dtype), xg, yg).permute(0, 3, 1, 2)
            cam_bwd = self._se2_disp(th_p, v_p, torch.ones(B, H, W, device=device, dtype=dtype), xg, yg).permute(0, 3, 1, 2)
            
            def disp_from_s(s_map):
                s_pos = s_map.clamp(min=0, max=1)
                s_neg = (-s_map).clamp(min=0, max=1)
                disp_pos = self._se2_disp(th_n, v_n, s_pos, xg, yg)
                disp_neg = self._se2_disp(th_p, v_p, s_neg, xg, yg)
                m = (s_map >= 0).float().unsqueeze(-1)
                return disp_pos * m + disp_neg * (1 - m)
        
        profile = getattr(params, "trajectory_profile", "constant")
        traj = []
        for i, t in enumerate(time_grid):
            t_scaled = t * params.shutter_length
            if profile == "constant" and params.camera_jerk > 0:
                t_scaled = t_scaled * (1 + params.camera_jerk * t_scaled ** 2)
            t_eff = (t_scaled + row_offset).expand(B, H, W)
            
            if profile == "acceleration":
                s_map = t_eff + 0.5 * getattr(params, "camera_acceleration", 0.0) * t_eff ** 2
            else:
                s_map = t_eff
            
            disp = disp_from_s(s_map)
            lateral = getattr(params, "lateral_acceleration", 0.0)
            if lateral != 0:
                disp_norm = torch.norm(disp, dim=-1, keepdim=True).clamp(min=1e-8)
                direction = disp / disp_norm
                perp = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
                disp = disp + lateral * 0.5 * t_eff.unsqueeze(-1) ** 2 * perp
            traj.append(disp)
        
        return torch.stack(traj, dim=1), cam_fwd, cam_bwd
    
    def _se2_inverse(self, T):
        R, t = T[:, :2, :2], T[:, :2, 2:3]
        Rt = R.transpose(1, 2)
        Tout = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
        Tout[:, :2, :2] = Rt
        Tout[:, :2, 2:3] = -Rt @ t
        return Tout
    
    def _se2_log(self, T):
        R, t = T[:, :2, :2], T[:, :2, 2]
        theta = torch.atan2(R[:, 1, 0], R[:, 0, 0]).view(-1, 1, 1)
        x2 = theta * theta
        A = torch.where(theta.abs() < 1e-6, 1 - x2/6, torch.sin(theta) / theta)
        B = torch.where(theta.abs() < 1e-6, theta/2, (1 - torch.cos(theta)) / theta)
        det = (A * A + B * B).view(-1, 1)
        inv00, inv01 = A.view(-1, 1) / det, B.view(-1, 1) / det
        vx = inv00[:, 0] * t[:, 0] + inv01[:, 0] * t[:, 1]
        vy = -inv01[:, 0] * t[:, 0] + inv00[:, 0] * t[:, 1]
        return theta, torch.stack([vx, vy], dim=1)
    
    def _se2_disp(self, theta, v, s, xg, yg):
        B, H, W = s.shape
        alpha = s * theta.view(B, 1, 1)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        x2 = alpha * alpha
        A = torch.where(alpha.abs() < 1e-6, 1 - x2/6, torch.sin(alpha) / alpha)
        Bm = torch.where(alpha.abs() < 1e-6, alpha/2, (1 - torch.cos(alpha)) / alpha)
        vx, vy = v[:, 0].view(B, 1, 1), v[:, 1].view(B, 1, 1)
        tx = A * vx - Bm * vy
        ty = Bm * vx + A * vy
        x = xg.view(1, H, W).expand(B, -1, -1)
        y = yg.view(1, H, W).expand(B, -1, -1)
        return torch.stack([ca * x - sa * y + tx - x, sa * x + ca * y + ty - y], dim=-1)
