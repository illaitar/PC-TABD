"""BlurGen: Physically-correct blur synthesis for deblurring datasets.

Main components:
- core: Blur synthesis engine (BlurConfig, BlurEngine)
- generators: CSV config and dataset generation
- codecs: Trajectory encoding/decoding
"""

from .core import (
    BlurConfig,
    BlurParams,
    BlurEngine,
    integrate_shutter,
    srgb_to_linear,
    linear_to_srgb,
    camera_motion,
    object_motion,
    build_trajectory,
    compute_visibility,
    extract_homography,
    extract_se2,
    homography_to_flow,
    extract_object_masks,
)
from .generators import CSVGenerator, DatasetGenerator
from .codecs import TrajectoryCodec

__version__ = "0.1.0"
__all__ = [
    # Core
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
    # Generators
    "CSVGenerator",
    "DatasetGenerator",
    # Codecs
    "TrajectoryCodec",
]
