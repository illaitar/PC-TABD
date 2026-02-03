"""SIN3D++ Visibility and Occlusion.

P4: Depth-ordered soft z-buffer w_vis = exp(-(z_src - zmin)/tau).

Fix (4): Normalize depth by robust p95 (not max); tau in normalized-depth units.
Fix (2): Optional forward-splat depth for zmin (many-to-one); else min-over-time.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def _warp_depth(depth: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Warp depth by disp; [B,1,H,W], [B,H,W,2] -> [B,1,H,W]."""
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing="ij")
    x = x.unsqueeze(0).expand(B, -1, -1) + disp[..., 0]
    y = y.unsqueeze(0).expand(B, -1, -1) + disp[..., 1]
    grid = torch.stack([2.0 * (x + 0.5) / W - 1.0, 2.0 * (y + 0.5) / H - 1.0], dim=-1)
    return F.grid_sample(depth, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _depth_scale_p95(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fix (4): Robust scale = p95 per batch. Avoids tying tau to max (scene-specific)."""
    B = depth.shape[0]
    flat = depth.reshape(B, -1)
    p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    p95 = p95.reshape(B, 1, 1, 1).clamp(min=eps)
    return p95


def _zmin_forward_splat(
    traj: torch.Tensor,  # [B, N, H, W, 2]
    z: torch.Tensor,  # [B, 1, H, W] normalized depth
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fix (2): Forward-splat z -> zmin at each ref over all contributors. ref = source - disp(source)."""
    B, N, H, W, _ = traj.shape
    z_flat = z.reshape(B, -1)
    zmin_flat = torch.full((B, H * W), float("inf"), device=device, dtype=dtype)
    yg = torch.arange(H, device=device, dtype=dtype)
    xg = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yg, xg, indexing="ij")

    for i in range(N):
        disp = traj[:, i]  # [B, H, W, 2]
        src_x = xx.unsqueeze(0) + disp[..., 0]
        src_y = yy.unsqueeze(0) + disp[..., 1]
        grid_src = torch.stack([
            2.0 * (src_x + 0.5) / W - 1.0,
            2.0 * (src_y + 0.5) / H - 1.0,
        ], dim=-1)
        disp_s = F.grid_sample(
            disp.permute(0, 3, 1, 2),
            grid_src,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [B, H, W, 2]
        ref_x = (src_x - disp_s[..., 0]).reshape(B, -1)
        ref_y = (src_y - disp_s[..., 1]).reshape(B, -1)
        rx = ref_x.round().long().clamp(0, W - 1)
        ry = ref_y.round().long().clamp(0, H - 1)
        idx = ry * W + rx  # [B, H*W]
        zmin_flat.scatter_reduce_(1, idx, z_flat, reduce="amin", include_self=True)

    zmin = zmin_flat.reshape(B, 1, H, W)
    inf_mask = torch.isinf(zmin)
    zmin = torch.where(inf_mask, z, zmin)
    return zmin


def compute_visibility(
    traj: torch.Tensor,  # [B, N, H, W, 2]
    depth: torch.Tensor,  # [B, 1, H, W]
    params,
    occ_mask: Optional[torch.Tensor] = None,
    use_depth_ordering: bool = True,
    visibility_tau: float = 0.03,
    use_forward_splat_zmin: bool = False,  # Fix (2): true z-buffer over contributors
) -> torch.Tensor:
    """Soft z-buffer w_vis = exp(-(z_src - zmin)/tau). Fix (4): depth norm by p95; Fix (2): optional forward-splat zmin.
    visibility_floor (on params): min visibility to avoid black borders at depth edges (default 0.12)."""
    B, N, H, W, _ = traj.shape
    device = traj.device
    dtype = traj.dtype
    visibility = torch.ones(B, N, H, W, device=device, dtype=dtype)

    if use_depth_ordering and depth is not None:
        # Fix (4): robust norm by p95; tau in normalized-depth units
        scale = _depth_scale_p95(depth)
        z = depth / (scale + 1e-8)
        z = z.clamp(0.0, 10.0)

        if use_forward_splat_zmin:
            # Fix (2): zmin at ref over all contributors (forward splat)
            zmin = _zmin_forward_splat(traj, z, device, dtype)
        else:
            depths = []
            for i in range(N):
                d = _warp_depth(z, traj[:, i])
                depths.append(d)
            stack = torch.stack(depths, dim=0)
            zmin = stack.amin(dim=0)

        tau = max(visibility_tau, 1e-6)
        # Min visibility floor: at depth boundaries bilinear warp mixes fg/bg depth → thin band of vis≈0 → black border. Floor avoids pure black.
        min_visibility = max(0.0, getattr(params, "visibility_floor", 0.12))
        for i in range(N):
            if use_forward_splat_zmin:
                z_i = _warp_depth(z, traj[:, i])
            else:
                z_i = depths[i]
            vis_i = torch.exp(-((z_i - zmin).clamp(min=0) / tau)).clamp(min_visibility, 1)
            visibility[:, i] = vis_i.squeeze(1)

    if occ_mask is not None:
        if occ_mask.dim() == 3:
            occ_mask = occ_mask.unsqueeze(1)
        occ_mask = occ_mask.expand(B, N, H, W)
        visibility = visibility * (1.0 - occ_mask)

    if params.occlusion_softness > 0:
        kernel_size = int(params.occlusion_softness * 10) + 1
        if kernel_size > 1 and kernel_size % 2 == 1:
            sigma = params.occlusion_softness * 5.0
            visibility = _gaussian_blur_2d(visibility, kernel_size, sigma)

    visibility = torch.clamp(visibility, 0.0, 1.0)
    return visibility


def _gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords = coords - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    B, N, H, W = x.shape
    x_flat = x.view(B * N, 1, H, W)
    x_blurred = F.conv2d(x_flat, kernel, padding=kernel_size // 2)
    return x_blurred.view(B, N, H, W)


def disocclusion_hallucination(
    warped: torch.Tensor,
    visibility: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    vis_expanded = visibility.unsqueeze(1)
    return warped * vis_expanded + original * (1.0 - vis_expanded)
