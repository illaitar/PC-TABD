"""SIN3D++ ISP: noise in linear/raw-like, sharpening post-ISP (P5)."""

import torch
import torch.nn.functional as F
from typing import Optional

from .shutter import linear_to_srgb


def apply_isp(
    blur: torch.Tensor,  # [B, 3, H, W] or [3, H, W]; linear if use_linear_input else sRGB
    params,
    use_linear_input: bool = False,  # P5: blur is linear; add Poisson+Gaussian, then linear_to_srgb, then sharpen
) -> torch.Tensor:
    """Apply ISP: P5 noise in linear (Poisson+Gaussian), linear_to_srgb, post-ISP sharpening.

    When use_linear_input: noise in linear -> linear_to_srgb -> sharpen.
    Otherwise: assume sRGB, Gaussian noise only (legacy), then sharpen.
    """
    single_batch = False
    if blur.dim() == 3:
        blur = blur.unsqueeze(0)
        single_batch = True

    processed = blur.clone()

    # P5: Noise in linear or raw-like space
    if use_linear_input:
        # Poisson-like: sqrt(L) * scale * N(0,1); Gaussian: sigma * N(0,1)
        poisson_scale = getattr(params, "noise_poisson_scale", 0.0)
        if poisson_scale > 0:
            L = processed.clamp(min=1e-6)
            processed = processed + torch.sqrt(L) * poisson_scale * torch.randn_like(processed)
        if params.noise_level > 0:
            processed = processed + params.noise_level * torch.randn_like(processed)
        processed = processed.clamp(min=0.0)
        processed = linear_to_srgb(processed)
    else:
        if params.noise_level > 0:
            processed = processed + params.noise_level * torch.randn_like(processed)

    # Post-ISP sharpening (after integration, "phone-like")
    if params.motion_sharpening > 0:
        gray = 0.299 * processed[:, 0] + 0.587 * processed[:, 1] + 0.114 * processed[:, 2]
        gray = gray.unsqueeze(1)
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            device=processed.device,
            dtype=processed.dtype,
        ).view(1, 1, 3, 3)
        edges = F.conv2d(gray, kernel, padding=1).abs()
        sharpening = params.motion_sharpening * edges.expand_as(processed)
        processed = processed + sharpening * (processed - F.avg_pool2d(processed, 3, 1, 1))

    processed = processed.clamp(0.0, 1.0)

    if single_batch:
        processed = processed.squeeze(0)
    return processed
