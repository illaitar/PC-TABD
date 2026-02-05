"""ISP: noise and sharpening post-processing."""

import torch
import torch.nn.functional as F

from .shutter import linear_to_srgb


def apply_isp(blur: torch.Tensor, params, use_linear_input: bool = False) -> torch.Tensor:
    """Apply ISP: noise in linear space, then sRGB conversion, then sharpening.
    
    Args:
        blur: [B, 3, H, W] or [3, H, W] blurred image
        params: BlurParams
        use_linear_input: blur is in linear space
    
    Returns:
        processed: [B, 3, H, W] or [3, H, W]
    """
    single = blur.dim() == 3
    if single:
        blur = blur.unsqueeze(0)
    
    out = blur.clone()
    
    if use_linear_input:
        # Poisson noise
        poisson = getattr(params, "noise_poisson_scale", 0.0)
        if poisson > 0:
            out = out + torch.sqrt(out.clamp(min=1e-6)) * poisson * torch.randn_like(out)
        
        # Gaussian noise
        if params.noise_level > 0:
            out = out + params.noise_level * torch.randn_like(out)
        
        out = linear_to_srgb(out.clamp(min=0))
    else:
        if params.noise_level > 0:
            out = out + params.noise_level * torch.randn_like(out)
    
    # Sharpening
    if params.motion_sharpening > 0:
        gray = 0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=out.device, dtype=out.dtype)
        edges = F.conv2d(gray.unsqueeze(1), kernel.view(1, 1, 3, 3), padding=1).abs()
        sharpening = params.motion_sharpening * edges.expand_as(out)
        out = out + sharpening * (out - F.avg_pool2d(out, 3, 1, 1))
    
    out = out.clamp(0, 1)
    return out.squeeze(0) if single else out
