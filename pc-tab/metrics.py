"""SSIM and LPIPS metrics for SIN3D evaluation."""

import torch
import torch.nn.functional as F
from typing import Optional

try:
    from lpips import LPIPS
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    # Don't print warning here, let the caller handle it


def ssim(
    img1: torch.Tensor,  # [B, C, H, W] or [C, H, W]
    img2: torch.Tensor,  # [B, C, H, W] or [C, H, W]
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM (Structural Similarity Index).
    
    Args:
        img1, img2: Images in range [0, 1]
        window_size: Size of Gaussian window
        C1, C2: Constants for numerical stability
        
    Returns:
        ssim: SSIM score [0, 1] (higher is better)
    """
    # Normalize dimensions
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    B, C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, device=device, dtype=dtype)
        coords = coords - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g[:, None] * g[None, :]
        return window.view(1, 1, size, size)
    
    window = gaussian_window(window_size).expand(C, 1, -1, -1)
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Average over spatial dimensions
    ssim = ssim_map.mean(dim=(1, 2, 3))
    
    return ssim.mean()  # Average over batch


class LPIPSMetric:
    """LPIPS (Learned Perceptual Image Patch Similarity) metric."""
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """Initialize LPIPS metric.
        
        Args:
            net: Network architecture ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        if not HAS_LPIPS:
            raise ImportError("LPIPS not available. Install with: pip install lpips")
        
        self.device = device
        self.lpips = LPIPS(net=net).to(device)
        self.lpips.eval()
    
    @torch.no_grad()
    def __call__(
        self,
        img1: torch.Tensor,  # [B, C, H, W] in [0, 1]
        img2: torch.Tensor,  # [B, C, H, W] in [0, 1]
    ) -> torch.Tensor:
        """Compute LPIPS distance.
        
        Args:
            img1, img2: Images in range [0, 1]
            
        Returns:
            lpips: LPIPS distance (lower is better)
        """
        # Normalize to [-1, 1] for LPIPS
        img1_norm = img1 * 2.0 - 1.0
        img2_norm = img2 * 2.0 - 1.0
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_val = self.lpips(img1_norm, img2_norm)
        
        return lpips_val.mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute PSNR.
    
    Args:
        img1, img2: Images in range [0, max_val]
        max_val: Maximum pixel value
        
    Returns:
        psnr: PSNR in dB (higher is better)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=img1.device)
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val
