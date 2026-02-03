"""P7: Regularizers on motion/warp (not on final image).

- L_inv: cycle consistency
- L_smooth: edge-aware spatial
- L_temp: acceleration prior
- L_fb: forward-backward consistency

Use when fine-tuning residual, calibrating depth-scale, or training motion heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def cycle_consistency_loss(
    flow_fwd: torch.Tensor,  # [B, 2, H, W] or [B, H, W, 2]
    flow_bwd: torch.Tensor,
    robust: bool = True,
    delta: float = 0.1,
) -> torch.Tensor:
    """L_inv: cycle consistency. flow_fwd(u) + flow_bwd(u + flow_fwd(u)) ≈ 0."""
    if flow_fwd.dim() == 4 and flow_fwd.shape[1] == 2:
        flow_fwd = flow_fwd.permute(0, 2, 3, 1)
        flow_bwd = flow_bwd.permute(0, 2, 3, 1)
    B, H, W, _ = flow_fwd.shape
    device = flow_fwd.device
    dtype = flow_fwd.dtype

    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    u_fwd = base + flow_fwd
    u_fwd_n = 2.0 * (u_fwd[..., 0] / (W - 1) - 0.5)
    v_fwd_n = 2.0 * (u_fwd[..., 1] / (H - 1) - 0.5)
    grid = torch.stack([u_fwd_n, v_fwd_n], dim=-1)

    flow_bwd_warped = F.grid_sample(
        flow_bwd.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    residual = flow_fwd + flow_bwd_warped
    err = (residual ** 2).sum(dim=-1)
    if robust:
        err = err / (err + delta ** 2)
    return err.mean()


def edge_aware_smooth_loss(
    motion: torch.Tensor,  # [B, 2, H, W] or [B, N, H, W, 2]
    edge_weight: Optional[torch.Tensor] = None,  # [B, 1, H, W] or [B, H, W]; high = edge
) -> torch.Tensor:
    """L_smooth: edge-aware spatial smoothness. Penalize motion gradients, downweight at edges."""
    if motion.dim() == 5:
        motion = motion.view(-1, motion.size(2), motion.size(3), motion.size(4))
        motion = motion.permute(0, 3, 1, 2)
    if motion.dim() == 4 and motion.shape[-1] == 2:
        motion = motion.permute(0, 3, 1, 2)

    B, C, H, W = motion.shape
    dx = motion[..., :, 1:] - motion[..., :, :-1]
    dy = motion[..., 1:, :] - motion[..., :-1, :]

    w_x = torch.ones(B, 1, H, W - 1, device=motion.device, dtype=motion.dtype)
    w_y = torch.ones(B, 1, H - 1, W, device=motion.device, dtype=motion.dtype)
    if edge_weight is not None:
        if edge_weight.dim() == 3:
            edge_weight = edge_weight.unsqueeze(1)
        ew = edge_weight
        w_x = (1.0 - ew[..., :, :-1]).clamp(min=0.01)
        w_y = (1.0 - ew[..., :-1, :]).clamp(min=0.01)

    loss_x = (w_x * (dx ** 2)).mean()
    loss_y = (w_y * (dy ** 2)).mean()
    return loss_x + loss_y


def acceleration_prior_loss(traj: torch.Tensor) -> torch.Tensor:
    """L_temp: penalize second temporal derivative of trajectory (smooth acceleration)."""
    if traj.dim() != 5:
        raise ValueError("traj must be [B, N, H, W, 2]")
    if traj.size(1) < 3:
        return traj.new_tensor(0.0)
    d2 = traj[:, 2:] - 2.0 * traj[:, 1:-1] + traj[:, :-2]
    return (d2 ** 2).mean()


def forward_backward_consistency_loss(
    flow_fwd: torch.Tensor,  # [B, 2, H, W]
    flow_bwd: torch.Tensor,
    robust: bool = True,
    delta: float = 0.1,
) -> torch.Tensor:
    """L_fb: forward-backward consistency. flow_fwd(u) ≈ -flow_bwd(u + flow_fwd(u))."""
    return cycle_consistency_loss(flow_fwd, flow_bwd, robust=robust, delta=delta)


class MotionRegularizers(nn.Module):
    """Combined P7 regularizers on motion/warp."""

    def __init__(
        self,
        lambda_inv: float = 0.1,
        lambda_smooth: float = 0.05,
        lambda_temp: float = 0.05,
        lambda_fb: float = 0.1,
        robust_cycle: bool = True,
        delta_cycle: float = 0.1,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_smooth = lambda_smooth
        self.lambda_temp = lambda_temp
        self.lambda_fb = lambda_fb
        self.robust_cycle = robust_cycle
        self.delta_cycle = delta_cycle

    def forward(
        self,
        traj: Optional[torch.Tensor] = None,  # [B, N, H, W, 2]
        flow_fwd: Optional[torch.Tensor] = None,
        flow_bwd: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        ref = flow_fwd if flow_fwd is not None else (traj if traj is not None else None)
        if ref is None:
            return torch.tensor(0.0, device="cpu"), {"total": torch.tensor(0.0, device="cpu")}
        total = ref.new_tensor(0.0)
        out = {}

        # L_fb / L_inv: same cycle-consistency penalty (forward-backward)
        if flow_fwd is not None and flow_bwd is not None and (self.lambda_fb > 0 or self.lambda_inv > 0):
            L_fb = forward_backward_consistency_loss(
                flow_fwd, flow_bwd, robust=self.robust_cycle, delta=self.delta_cycle
            )
            out["L_fb"] = out["L_inv"] = L_fb
            total = total + (self.lambda_fb + self.lambda_inv) * L_fb

        if traj is not None and self.lambda_smooth > 0:
            L_smooth = edge_aware_smooth_loss(traj, edge_weight=edge_weight)
            out["L_smooth"] = L_smooth
            total = total + self.lambda_smooth * L_smooth

        if traj is not None and traj.size(1) >= 3 and self.lambda_temp > 0:
            L_temp = acceleration_prior_loss(traj)
            out["L_temp"] = L_temp
            total = total + self.lambda_temp * L_temp

        out["total"] = total
        return total, out
