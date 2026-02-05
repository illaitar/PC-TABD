"""Shutter integration: exposure accumulation along trajectory."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .trajectories import get_rs_time_offset


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to linear light."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert linear light [0,1] to sRGB."""
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1/2.4)) - 0.055)


def _soften_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Apply Gaussian blur for soft mask edges."""
    if feather <= 0:
        return mask
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] > 1:
        mask = mask.sum(dim=1, keepdim=True).clamp(0, 1)
    
    k = min(feather * 2 + 1, 15)
    if k < 3:
        return mask
    
    sigma = feather / 2.0
    khalf = k // 2
    x = torch.arange(-khalf, khalf + 1, device=mask.device, dtype=mask.dtype)
    g = torch.exp(-x**2 / (2 * sigma**2))
    g = g / g.sum()
    
    # Use same padding as kernel half
    m = F.pad(mask, (khalf, khalf, khalf, khalf), mode='replicate')
    m = F.conv2d(m, g.view(1, 1, -1, 1), padding=0)
    m = F.conv2d(m, g.view(1, 1, 1, -1), padding=0)
    return m.clamp(0, 1)


def _inpaint_diffusion(img: torch.Tensor, hole: torch.Tensor, iters: int = 25, k: int = 3) -> torch.Tensor:
    """Fill holes via diffusion. img [B,C,H,W], hole [B,1,H,W] (1=fill)."""
    if hole.dim() == 3:
        hole = hole.unsqueeze(1)
    B, C, H, W = img.shape
    
    # Ensure hole has same spatial size as img
    if hole.shape[2] != H or hole.shape[3] != W:
        hole = F.interpolate(hole, size=(H, W), mode='nearest')
    
    hole = hole.clamp(0, 1).expand(-1, C, -1, -1)
    known = 1.0 - hole
    
    kernel = torch.ones(C, 1, k, k, device=img.device, dtype=img.dtype)
    kernel_1 = torch.ones(1, 1, k, k, device=img.device, dtype=img.dtype)
    pad = k // 2
    
    x = img.clone()
    for _ in range(iters):
        num = F.conv2d(x * known, kernel, padding=pad, groups=C)
        den = F.conv2d(known[:, :1], kernel_1, padding=pad)
        den = den.clamp(min=1e-6).expand(-1, C, -1, -1)
        x = x * known + (num / den) * hole
    return x


def integrate_shutter(
    sharp_seq: torch.Tensor,
    traj: torch.Tensor,
    visibility: torch.Tensor,
    params,
    use_linear_light: bool = True,
    use_zbuffer: bool = False,
    return_linear: bool = False,
    cam_traj: Optional[torch.Tensor] = None,
    obj_traj: Optional[torch.Tensor] = None,
    object_masks: Optional[torch.Tensor] = None,
    object_mask_feather: int = 3,
    bg_fill: str = "inpaint",
    object_confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrate exposure along trajectory.
    
    Args:
        sharp_seq: [B,T,3,H,W] sharp frames in [0,1]
        traj: [B,N,H,W,2] displacement field
        visibility: [B,N,H,W] visibility weights
        params: BlurParams
        use_linear_light: integrate in linear space
        use_zbuffer: skip weight normalization (z-buffer mode)
        return_linear: return linear output (for ISP)
        cam_traj: [B,N,H,W,2] camera trajectory (for layered)
        obj_traj: [B,N,H,W,2] object trajectory (for layered)
        object_masks: [B,K,H,W] object masks
        object_mask_feather: soft mask feather pixels
        bg_fill: "inpaint" or "neighbors"
        object_confidence: [B,1,H,W] soft object confidence for smooth blending
    
    Returns:
        blur: [B,3,H,W] blurred image
    """
    single_batch = sharp_seq.dim() == 4
    if single_batch:
        sharp_seq = sharp_seq.unsqueeze(0)
    
    B, T, C, H, W = sharp_seq.shape
    N = traj.shape[1]
    device, dtype = traj.device, traj.dtype
    center_idx = T // 2
    
    use_layered = (
        object_masks is not None
        and object_masks.numel() > 0
        and object_masks.shape[1] > 0
        and cam_traj is not None
        and obj_traj is not None
    )
    
    if sharp_seq.min() < 0:
        sharp_seq = (sharp_seq + 1.0) / 2.0
    
    sharp_linear = srgb_to_linear(sharp_seq) if use_linear_light else sharp_seq
    
    gain = getattr(params, "exposure_gain", 1.0)
    if gain != 1.0:
        sharp_linear = sharp_linear * gain
    
    time_result = params.get_time_grid()
    if isinstance(time_result, tuple):
        time_grid, weights = time_result
    else:
        time_grid = time_result
        weights = torch.ones_like(time_grid) / len(time_grid)
    weights = (weights / weights.sum()).to(device)
    
    blur = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    center_linear = sharp_linear[:, center_idx]
    
    yg, xg = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    
    if use_layered:
        # Use soft confidence for smooth trajectory blending if available
        if object_confidence is not None:
            # Soft blending: blend trajectories based on confidence
            # trajectory = camera + confidence * object_offset
            # This creates smooth transitions at object boundaries
            conf = object_confidence  # [B, 1, H, W]
            conf_feather = _soften_mask(conf, object_mask_feather)
            
            for i in range(N):
                disp_c = cam_traj[:, i]  # [B, H, W, 2]
                disp_o = obj_traj[:, i]  # [B, H, W, 2]
                vis = visibility[:, i:i+1]
                w = weights[i].view(1, 1, 1, 1) * vis
                
                # Warp confidence with object trajectory to get time-dependent blend
                disp_o_2d = disp_o.permute(0, 3, 1, 2)  # [B, 2, H, W]
                xo = xg.unsqueeze(0).expand(B, -1, -1) + disp_o[..., 0]
                yo = yg.unsqueeze(0).expand(B, -1, -1) + disp_o[..., 1]
                grid_o = torch.stack([2*(xo+0.5)/W - 1, 2*(yo+0.5)/H - 1], dim=-1)
                conf_t = F.grid_sample(conf_feather, grid_o, mode="bilinear", padding_mode="zeros", align_corners=False)
                
                # Soft blend: trajectory = camera + conf * object
                conf_exp = conf_t.permute(0, 2, 3, 1)  # [B, H, W, 1]
                disp_blend = disp_c + conf_exp * disp_o
                
                # Warp image with blended trajectory
                xd = xg.unsqueeze(0).expand(B, -1, -1) + disp_blend[..., 0]
                yd = yg.unsqueeze(0).expand(B, -1, -1) + disp_blend[..., 1]
                grid_d = torch.stack([2*(xd+0.5)/W - 1, 2*(yd+0.5)/H - 1], dim=-1)
                warped = F.grid_sample(center_linear, grid_d, mode="bilinear", padding_mode="border", align_corners=False)
                
                blur = blur + warped * w
                weight_sum = weight_sum + w
        else:
            # Binary mask mode: separate bg and object rendering
            M0 = (object_masks.sum(dim=1, keepdim=True) > 0).float()
            M0_soft = _soften_mask(M0, object_mask_feather)
            
            blur_bg = torch.zeros(B, C, H, W, device=device, dtype=dtype)
            blur_obj = torch.zeros(B, C, H, W, device=device, dtype=dtype)
            weight_bg = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            weight_obj = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            alpha_acc = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            
            for i in range(N):
                disp_c = cam_traj[:, i]
                disp_o = obj_traj[:, i]
                vis = visibility[:, i:i+1]
                w = weights[i].view(1, 1, 1, 1) * vis
                
                # Warp background with camera trajectory
                xc = xg.unsqueeze(0).expand(B, -1, -1) + disp_c[..., 0]
                yc = yg.unsqueeze(0).expand(B, -1, -1) + disp_c[..., 1]
                grid_c = torch.stack([2*(xc+0.5)/W - 1, 2*(yc+0.5)/H - 1], dim=-1)
                warped_bg = F.grid_sample(center_linear, grid_c, mode="bilinear", padding_mode="border", align_corners=False)
                
                # Warp object with camera + object trajectory
                disp_total = disp_c + disp_o
                xo = xg.unsqueeze(0).expand(B, -1, -1) + disp_total[..., 0]
                yo = yg.unsqueeze(0).expand(B, -1, -1) + disp_total[..., 1]
                grid_o = torch.stack([2*(xo+0.5)/W - 1, 2*(yo+0.5)/H - 1], dim=-1)
                warped_obj = F.grid_sample(center_linear, grid_o, mode="bilinear", padding_mode="border", align_corners=False)
                
                # Time-dependent object mask
                alpha_t = F.grid_sample(M0_soft, grid_o, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)
                
                blur_bg = blur_bg + warped_bg * w
                blur_obj = blur_obj + warped_obj * w
                weight_bg = weight_bg + w
                weight_obj = weight_obj + w
                alpha_acc = alpha_acc + alpha_t * w
            
            blur_bg = blur_bg / weight_bg.clamp(min=1e-6)
            blur_obj = blur_obj / weight_obj.clamp(min=1e-6)
            alpha_final = (alpha_acc / weight_bg.clamp(min=1e-6)).clamp(0, 1)
            blur = alpha_final * blur_obj + (1 - alpha_final) * blur_bg
            weight_sum = weight_bg
    else:
        for i in range(N):
            disp = traj[:, i]
            xc = xg.unsqueeze(0).expand(B, -1, -1) + disp[..., 0]
            yc = yg.unsqueeze(0).expand(B, -1, -1) + disp[..., 1]
            grid = torch.stack([2*(xc+0.5)/W - 1, 2*(yc+0.5)/H - 1], dim=-1)
            warped = F.grid_sample(center_linear, grid, mode="bilinear", padding_mode="border", align_corners=False)
            
            vis = visibility[:, i:i+1]
            w = weights[i].view(1, 1, 1, 1) * vis
            blur = blur + warped * w
            weight_sum = weight_sum + w
    
    if not use_zbuffer:
        blur = blur / weight_sum.clamp(min=1e-6)
    else:
        blend = (weight_sum > 1e-4).float()
        blur = blur * blend + center_linear * (1 - blend)
    
    if use_linear_light and not return_linear:
        blur = linear_to_srgb(blur)
    
    if single_batch:
        blur = blur.squeeze(0)
    
    return blur
