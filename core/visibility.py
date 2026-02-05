"""Visibility computation with depth-ordered soft z-buffer."""

import torch
import torch.nn.functional as F
from typing import Optional


def _warp_depth(depth: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Warp depth by displacement field."""
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype), indexing="ij")
    x = x.unsqueeze(0).expand(B, -1, -1) + disp[..., 0]
    y = y.unsqueeze(0).expand(B, -1, -1) + disp[..., 1]
    grid = torch.stack([2*(x+0.5)/W - 1, 2*(y+0.5)/H - 1], dim=-1)
    return F.grid_sample(depth, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _depth_scale_p95(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust scale = p95 per batch."""
    B = depth.shape[0]
    flat = depth.reshape(B, -1)
    p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True).reshape(B, 1, 1, 1)
    return p95.clamp(min=eps)


def compute_visibility(
    traj: torch.Tensor,
    depth: torch.Tensor,
    params,
    occ_mask: Optional[torch.Tensor] = None,
    use_depth_ordering: bool = True,
    visibility_tau: float = 0.03,
) -> torch.Tensor:
    """Compute soft z-buffer visibility.
    
    Args:
        traj: [B, N, H, W, 2] trajectory field
        depth: [B, 1, H, W] depth map
        params: BlurParams
        occ_mask: [B, 1, H, W] optional occlusion mask
        use_depth_ordering: use depth-based visibility
        visibility_tau: z-buffer softness
    
    Returns:
        visibility: [B, N, H, W]
    """
    B, N, H, W, _ = traj.shape
    device, dtype = traj.device, traj.dtype
    visibility = torch.ones(B, N, H, W, device=device, dtype=dtype)
    
    if use_depth_ordering and depth is not None:
        scale = _depth_scale_p95(depth)
        z = (depth / (scale + 1e-8)).clamp(0, 10)
        
        depths = [_warp_depth(z, traj[:, i]) for i in range(N)]
        zmin = torch.stack(depths, dim=0).amin(dim=0)
        
        tau = max(visibility_tau, 1e-6)
        min_vis = max(0.0, getattr(params, "visibility_floor", 0.12))
        
        for i in range(N):
            vis_i = torch.exp(-((depths[i] - zmin).clamp(min=0) / tau)).clamp(min_vis, 1)
            visibility[:, i] = vis_i.squeeze(1)
    
    if occ_mask is not None:
        if occ_mask.dim() == 3:
            occ_mask = occ_mask.unsqueeze(1)
        visibility = visibility * (1 - occ_mask.expand(B, N, H, W))
    
    if params.occlusion_softness > 0:
        k = int(params.occlusion_softness * 10) + 1
        if k > 1 and k % 2 == 1:
            visibility = _gaussian_blur_2d(visibility, k, params.occlusion_softness * 5)
    
    return visibility.clamp(0, 1)


def _gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    kernel = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    kernel = kernel / kernel.sum()
    B, N, H, W = x.shape
    x_flat = x.view(B * N, 1, H, W)
    return F.conv2d(x_flat, kernel, padding=kernel_size // 2).view(B, N, H, W)
