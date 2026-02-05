"""Blur synthesis configuration."""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
import math
import numpy as np
import torch


@dataclass
class BlurConfig:
    """Configuration for blur synthesis with parameter ranges.
    
    Each range parameter is (min, max) for uniform sampling.
    """
    # Camera motion
    camera_translation_scale: Tuple[float, float] = (0.5, 2.0)
    camera_rotation_scale: Tuple[float, float] = (0.0, 2 * math.pi)
    camera_jerk: Tuple[float, float] = (0.0, 0.05)
    rolling_shutter_strength: Tuple[float, float] = (0.0, 0.3)
    
    # Object motion - scales the difference between object and camera SE2
    object_scale_range: Tuple[float, float] = (0.8, 1.2)  # scale object motion offset
    object_direction_range: Tuple[float, float] = (0.0, 0.0)
    object_nonrigid_noise: Tuple[float, float] = (0.0, 0.1)
    object_model: str = "se2"  # "dense" | "se2"
    object_residual_threshold: float = 3.0  # residual < threshold → background
    object_saturation: float = 10.0  # residual >= saturation → full object confidence
    object_mask_feather: int = 3
    object_adaptive_threshold: bool = True
    object_soft_blend: bool = True  # soft blending for smooth transitions
    
    # Shutter
    shutter_length: Tuple[float, float] = (0.5, 2.0)
    shutter_profile: str = "box"  # "box" | "triangle" | "skewed" | "gaussian"
    shutter_skew: Tuple[float, float] = (0.3, 0.7)
    num_subframes: int = 32
    
    # Trajectory
    trajectory_profile: str = "acceleration"  # "constant" | "acceleration" | "smooth_walk"
    camera_acceleration: Tuple[float, float] = (-5.0, 5.0)
    lateral_acceleration: Tuple[float, float] = (-1.7, 1.7)
    smooth_walk_jerk: Tuple[float, float] = (0.0, 0.02)
    
    # Visibility
    occlusion_softness: Tuple[float, float] = (0.0, 0.1)
    visibility_floor: float = 0.12
    
    # ISP
    noise_level: Tuple[float, float] = (0.0, 0.02)
    noise_poisson_scale: Tuple[float, float] = (0.0, 0.01)
    motion_sharpening: Tuple[float, float] = (0.0, 0.1)
    exposure_gain: Tuple[float, float] = (0.9, 1.1)
    
    # Depth
    depth_parallax_scale: Tuple[float, float] = (0.5, 2.0)
    
    # Camera model
    camera_model: str = "se2_rigid"  # "homography" | "flow" | "se2_rigid"
    se2_mode: str = "symmetric"
    se2_rotation_scale: Tuple[float, float] = (1.0, 1.0)
    
    # Background fill
    bg_fill: str = "inpaint"  # "inpaint" | "neighbors"
    
    # Rolling shutter reference
    rs_y0: float = 0.0
    
    device: str = "cpu"
    
    def sample(self, rng: Optional[np.random.Generator] = None) -> "BlurParams":
        """Sample random parameters from config ranges."""
        if rng is None:
            rng = np.random.default_rng()
        
        def _uniform(r: Tuple[float, float]) -> float:
            return float(rng.uniform(r[0], r[1]))
        
        return BlurParams(
            camera_translation_scale=_uniform(self.camera_translation_scale),
            camera_rotation_scale=_uniform(self.camera_rotation_scale),
            camera_jerk=_uniform(self.camera_jerk),
            rolling_shutter_strength=_uniform(self.rolling_shutter_strength),
            object_scale=_uniform(self.object_scale_range),
            object_direction=_uniform(self.object_direction_range),
            object_nonrigid_noise=_uniform(self.object_nonrigid_noise),
            shutter_length=_uniform(self.shutter_length),
            shutter_profile=self.shutter_profile,
            shutter_skew=_uniform(self.shutter_skew),
            num_subframes=self.num_subframes,
            trajectory_profile=self.trajectory_profile,
            camera_acceleration=_uniform(self.camera_acceleration),
            lateral_acceleration=_uniform(self.lateral_acceleration),
            smooth_walk_jerk=_uniform(self.smooth_walk_jerk),
            occlusion_softness=_uniform(self.occlusion_softness),
            noise_level=_uniform(self.noise_level),
            noise_poisson_scale=_uniform(self.noise_poisson_scale),
            motion_sharpening=_uniform(self.motion_sharpening),
            exposure_gain=_uniform(self.exposure_gain),
            depth_parallax_scale=_uniform(self.depth_parallax_scale),
            se2_rotation_scale=_uniform(self.se2_rotation_scale),
            object_model=self.object_model,
            device=self.device,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlurConfig":
        return cls(**d)


@dataclass
class BlurParams:
    """Sampled parameters for a single blur synthesis."""
    camera_translation_scale: float
    camera_rotation_scale: float
    camera_jerk: float
    rolling_shutter_strength: float
    object_scale: float
    object_direction: float
    object_nonrigid_noise: float
    object_acceleration: float = 0.0  # separate from camera
    object_lateral_acceleration: float = 0.0  # separate from camera
    shutter_length: float = 1.0
    shutter_profile: str = "box"
    shutter_skew: float = 0.5
    num_subframes: int = 32
    trajectory_profile: str = "acceleration"
    camera_acceleration: float = 0.0
    lateral_acceleration: float = 0.0
    smooth_walk_jerk: float = 0.0
    occlusion_softness: float = 0.0
    noise_level: float = 0.0
    noise_poisson_scale: float = 0.0
    motion_sharpening: float = 0.0
    exposure_gain: float = 1.0
    depth_parallax_scale: float = 1.0
    se2_rotation_scale: float = 1.0
    object_model: str = "se2"
    device: str = "cpu"
    
    # Internal state (set by engine)
    _camera_flow_fwd: Optional[torch.Tensor] = field(default=None, repr=False)
    _camera_flow_bwd: Optional[torch.Tensor] = field(default=None, repr=False)
    _camera_from_pose: bool = field(default=False, repr=False)
    _cam_traj_se2: Optional[torch.Tensor] = field(default=None, repr=False)
    rs_y0: float = field(default=0.0, repr=False)
    object_residual_max_norm: float = field(default=2.0, repr=False)
    visibility_floor: float = field(default=0.12, repr=False)
    
    def get_time_grid(self) -> torch.Tensor:
        """Get normalized time grid [-1, 1] for shutter integration."""
        device = torch.device(self.device)
        t = torch.linspace(-1, 1, self.num_subframes, device=device)
        
        if self.shutter_profile == "box":
            return t
        elif self.shutter_profile == "triangle":
            w = 1.0 - torch.abs(t)
            return t, w / w.sum()
        elif self.shutter_profile == "skewed":
            if self.shutter_skew > 0.5:
                w = torch.exp(-self.shutter_skew * (t + 1) ** 2)
            else:
                w = torch.exp(-(1 - self.shutter_skew) * (t - 1) ** 2)
            return t, w / w.sum()
        elif self.shutter_profile == "gaussian":
            w = torch.exp(-0.5 * (t / 0.3) ** 2)
            return t, w / w.sum()
        return t
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in asdict(self).items() 
            if not k.startswith("_") and not isinstance(v, torch.Tensor)
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlurParams":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
