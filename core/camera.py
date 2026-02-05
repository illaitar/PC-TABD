"""Camera motion extraction and object segmentation."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import label, binary_fill_holes
from typing import Tuple, Optional


def extract_homography(
    img1: torch.Tensor,
    img2: torch.Tensor,
    ransac_threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract homography between two images using SIFT+RANSAC.
    
    Args:
        img1: [B, 3, H, W] or [3, H, W] reference image
        img2: same shape, target image
        ransac_threshold: RANSAC reprojection threshold
    
    Returns:
        H: [B, 3, 3] or [3, 3] homography matrix
        inlier_mask: [B, H, W] or [H, W]
    """
    single = img1.dim() == 3
    if single:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    
    B, C, H_img, W_img = img1.shape
    device, dtype = img1.device, img1.dtype
    
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    if img1_np.min() < 0:
        img1_np = (img1_np + 1) / 2
        img2_np = (img2_np + 1) / 2
    img1_np = (img1_np * 255).clip(0, 255).astype(np.uint8)
    img2_np = (img2_np * 255).clip(0, 255).astype(np.uint8)
    
    homographies, masks = [], []
    
    for b in range(B):
        if C == 3:
            gray1 = cv2.cvtColor(img1_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1_np[b, 0], img2_np[b, 0]
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 4:
            homographies.append(np.eye(3, dtype=np.float32))
            masks.append(np.ones((H_img, W_img), dtype=np.float32))
            continue
        
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        
        if len(good) < 4:
            homographies.append(np.eye(3, dtype=np.float32))
            masks.append(np.ones((H_img, W_img), dtype=np.float32))
            continue
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        homographies.append((H_mat if H_mat is not None else np.eye(3)).astype(np.float32))
        masks.append(np.ones((H_img, W_img), dtype=np.float32))
    
    H_tensor = torch.from_numpy(np.stack(homographies)).to(device=device, dtype=dtype)
    mask_tensor = torch.from_numpy(np.stack(masks)).to(device=device, dtype=dtype)
    
    if single:
        return H_tensor.squeeze(0), mask_tensor.squeeze(0)
    return H_tensor, mask_tensor


def extract_se2(
    img1: torch.Tensor,
    img2: torch.Tensor,
    ransac_threshold: float = 3.0,
) -> torch.Tensor:
    """Extract SE(2) transform (rotation + translation) using SIFT+RANSAC."""
    single = img1.dim() == 3
    if single:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    
    B, C, H, W = img1.shape
    device, dtype = img1.device, img1.dtype
    
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    if img1_np.min() < 0:
        img1_np = (img1_np + 1) / 2
        img2_np = (img2_np + 1) / 2
    img1_np = (img1_np * 255).clip(0, 255).astype(np.uint8)
    img2_np = (img2_np * 255).clip(0, 255).astype(np.uint8)
    
    Ts = []
    for b in range(B):
        if C == 3:
            gray1 = cv2.cvtColor(img1_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1_np[b, 0], img2_np[b, 0]
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 4:
            Ts.append(np.eye(3, dtype=np.float32))
            continue
        
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        
        if len(good) < 4:
            Ts.append(np.eye(3, dtype=np.float32))
            continue
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        A, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
        
        if A is None:
            Ts.append(np.eye(3, dtype=np.float32))
            continue
        
        # Project to pure rotation
        R = A[:, :2].astype(np.float32)
        t = A[:, 2].astype(np.float32)
        U, _, Vt = np.linalg.svd(R)
        R = (U @ Vt).astype(np.float32)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = (U @ Vt).astype(np.float32)
        
        T = np.eye(3, dtype=np.float32)
        T[:2, :2] = R
        T[:2, 2] = t
        Ts.append(T)
    
    T_tensor = torch.from_numpy(np.stack(Ts)).to(device=device, dtype=dtype)
    return T_tensor.squeeze(0) if single else T_tensor


def homography_to_flow(H: torch.Tensor, H_img: int, W_img: int) -> torch.Tensor:
    """Convert homography to dense displacement field."""
    single = H.dim() == 2
    if single:
        H = H.unsqueeze(0)
    
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    
    y, x = torch.meshgrid(torch.arange(H_img, device=device, dtype=dtype),
                          torch.arange(W_img, device=device, dtype=dtype), indexing="ij")
    coords = torch.stack([x.flatten(), y.flatten(), torch.ones(H_img * W_img, device=device, dtype=dtype)])
    coords = coords.unsqueeze(0).expand(B, -1, -1)
    
    transformed = torch.bmm(H, coords)
    z = transformed[:, 2:3].clamp(min=1e-6)
    transformed = transformed[:, :2] / z
    
    flow = transformed.view(B, 2, H_img, W_img)
    flow = flow - torch.stack([x, y]).unsqueeze(0).expand(B, -1, -1, -1)
    
    return flow.squeeze(0) if single else flow


def fit_se2_from_flow(
    flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ransac_threshold: float = 3.0,
) -> Optional[torch.Tensor]:
    """Fit SE(2) transform from optical flow using RANSAC.
    
    Args:
        flow: [2, H, W] optical flow
        mask: optional [H, W] binary mask for region fitting
        ransac_threshold: RANSAC reprojection threshold
    
    Returns:
        T: [3, 3] SE(2) transform matrix or None if failed
    """
    H, W = flow.shape[1], flow.shape[2]
    flow_np = flow.permute(1, 2, 0).cpu().numpy()
    
    if mask is not None:
        mask_np = mask.cpu().numpy() > 0.5
        if mask_np.sum() < 100:
            return None
        yy, xx = np.where(mask_np)
    else:
        # Sample grid points
        step = 10
        yy, xx = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing='ij')
        yy, xx = yy.flatten(), xx.flatten()
    
    pts_src = np.stack([xx, yy], axis=1).astype(np.float32)
    flow_at_pts = flow_np[yy, xx]
    pts_dst = pts_src + flow_at_pts
    
    if len(pts_src) < 10:
        return None
    
    if len(pts_src) > 5000:
        idx = np.random.choice(len(pts_src), 5000, replace=False)
        pts_src = pts_src[idx]
        pts_dst = pts_dst[idx]
    
    A, _ = cv2.estimateAffinePartial2D(
        pts_src.reshape(-1, 1, 2),
        pts_dst.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    
    if A is None:
        return None
    
    # Project to pure rotation
    R = A[:, :2].astype(np.float32)
    t = A[:, 2].astype(np.float32)
    U, _, Vt = np.linalg.svd(R)
    R = (U @ Vt).astype(np.float32)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = (U @ Vt).astype(np.float32)
    
    T = np.eye(3, dtype=np.float32)
    T[:2, :2] = R
    T[:2, 2] = t
    
    return torch.from_numpy(T).to(flow.device)


def se2_to_flow(T: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert SE(2) transform to dense flow field.
    
    Args:
        T: [3, 3] SE(2) transform
        H, W: image dimensions
    
    Returns:
        flow: [2, H, W] dense flow
    """
    device, dtype = T.device, T.dtype
    yg, xg = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    
    theta = torch.atan2(T[1, 0], T[0, 0])
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    tx, ty = T[0, 2], T[1, 2]
    
    new_x = cos_t * xg + sin_t * yg + tx
    new_y = -sin_t * xg + cos_t * yg + ty
    
    return torch.stack([new_x - xg, new_y - yg], dim=0)


def compute_object_confidence(
    residual_fwd: torch.Tensor,
    residual_bwd: Optional[torch.Tensor] = None,
    threshold: float = 2.0,
    saturation: float = 8.0,
) -> torch.Tensor:
    """Compute soft object confidence from residual flow magnitude.
    
    Returns smooth [0, 1] confidence: 0 at threshold, 1 at saturation.
    This creates physically plausible smooth transitions.
    
    Args:
        residual_fwd: [B, 2, H, W] forward residual flow
        residual_bwd: [B, 2, H, W] optional backward residual
        threshold: magnitude below which confidence is 0
        saturation: magnitude at which confidence is 1
    
    Returns:
        confidence: [B, 1, H, W] soft object mask in [0, 1]
    """
    if residual_fwd.dim() == 4 and residual_fwd.shape[-1] == 2:
        residual_fwd = residual_fwd.permute(0, 3, 1, 2)
    if residual_bwd is not None and residual_bwd.dim() == 4 and residual_bwd.shape[-1] == 2:
        residual_bwd = residual_bwd.permute(0, 3, 1, 2)
    
    B, _, H, W = residual_fwd.shape
    device, dtype = residual_fwd.device, residual_fwd.dtype
    
    mag_f = torch.norm(residual_fwd, dim=1, keepdim=True)
    if residual_bwd is not None:
        mag_b = torch.norm(residual_bwd, dim=1, keepdim=True)
        mag = torch.sqrt(0.5 * (mag_f**2 + mag_b**2))
    else:
        mag = mag_f
    
    # Smooth with average pooling
    mag = F.avg_pool2d(mag, 5, 1, 2)
    
    # Soft sigmoid-like transition: 0 below threshold, 1 above saturation
    # Using smooth step: 3x^2 - 2x^3 (Hermite interpolation)
    x = ((mag - threshold) / (saturation - threshold)).clamp(0, 1)
    confidence = x * x * (3 - 2 * x)
    
    return confidence


def extract_object_masks(
    residual_fwd: torch.Tensor,
    residual_bwd: Optional[torch.Tensor] = None,
    threshold: float = 2.0,
    min_area: int = 100,
    max_objects: int = 5,
    adaptive: bool = True,
    mad_mult: float = 3.0,
) -> torch.Tensor:
    """Extract object masks from residual flow.
    
    Args:
        residual_fwd: [B, 2, H, W] forward residual flow
        residual_bwd: [B, 2, H, W] optional backward residual
        threshold: motion magnitude threshold
        min_area: minimum component area
        max_objects: maximum objects to return
        adaptive: use adaptive thresholding
        mad_mult: MAD multiplier for adaptive threshold
    
    Returns:
        masks: [B, K, H, W] object masks
    """
    if residual_fwd.dim() == 4 and residual_fwd.shape[-1] == 2:
        residual_fwd = residual_fwd.permute(0, 3, 1, 2)
    if residual_bwd is not None and residual_bwd.dim() == 4 and residual_bwd.shape[-1] == 2:
        residual_bwd = residual_bwd.permute(0, 3, 1, 2)
    
    B, _, H, W = residual_fwd.shape
    device, dtype = residual_fwd.device, residual_fwd.dtype
    
    mag_f = torch.norm(residual_fwd, dim=1)
    if residual_bwd is not None:
        mag_b = torch.norm(residual_bwd, dim=1)
        mag = torch.sqrt(0.5 * (mag_f**2 + mag_b**2))
    else:
        mag = mag_f
    
    # Smooth
    mag = F.avg_pool2d(mag.unsqueeze(1), 5, 1, 2).squeeze(1)
    
    # Threshold
    thr = torch.full((B,), threshold, device=device, dtype=dtype)
    if adaptive:
        flat = mag.view(B, -1)
        med = flat.median(dim=1).values
        mad = (flat - med[:, None]).abs().median(dim=1).values
        thr = torch.maximum(thr, med + mad_mult * mad)
    
    obj = mag > thr.view(B, 1, 1)
    
    all_masks = []
    for b in range(B):
        m = obj[b].cpu().numpy().astype(np.uint8)
        if m.sum() < min_area:
            all_masks.append(torch.zeros(max_objects, H, W, device=device, dtype=dtype))
            continue
        
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = binary_fill_holes(m > 0).astype(np.uint8)
        
        labeled, num = label(m > 0)
        if num == 0:
            all_masks.append(torch.zeros(max_objects, H, W, device=device, dtype=dtype))
            continue
        
        mag_np = mag[b].cpu().numpy()
        comps = []
        for i in range(1, num + 1):
            comp = (labeled == i)
            area = comp.sum()
            if area < min_area:
                continue
            mean_mag = mag_np[comp].mean()
            comps.append((area * mean_mag, comp.astype(np.float32)))
        
        if not comps:
            all_masks.append(torch.zeros(max_objects, H, W, device=device, dtype=dtype))
            continue
        
        comps.sort(key=lambda x: x[0], reverse=True)
        masks_b = [torch.from_numpy(c).to(device=device, dtype=dtype) for _, c in comps[:max_objects]]
        while len(masks_b) < max_objects:
            masks_b.append(torch.zeros(H, W, device=device, dtype=dtype))
        all_masks.append(torch.stack(masks_b))
    
    return torch.stack(all_masks)
