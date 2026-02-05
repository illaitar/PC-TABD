"""BlurGen Core: Physically-correct blur synthesis engine."""

from .config import BlurConfig, BlurParams
from .engine import BlurEngine
from .shutter import integrate_shutter, srgb_to_linear, linear_to_srgb
from .trajectories import camera_motion, object_motion, build_trajectory
from .visibility import compute_visibility
from .camera import (
    extract_homography,
    extract_se2,
    homography_to_flow,
    extract_object_masks,
)

__all__ = [
    "BlurConfig",
    "BlurParams",
    "BlurEngine",
    "integrate_shutter",
    "srgb_to_linear",
    "linear_to_srgb",
    "camera_motion",
    "object_motion",
    "build_trajectory",
    "compute_visibility",
    "extract_homography",
    "extract_se2",
    "homography_to_flow",
    "extract_object_masks",
]
