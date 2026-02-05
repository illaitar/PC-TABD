"""Tests for BlurConfig and BlurParams."""

import pytest
import numpy as np
import torch

from blurgen.core import BlurConfig, BlurParams


class TestBlurConfig:
    def test_default_config(self):
        cfg = BlurConfig()
        assert cfg.shutter_length == (0.5, 2.0)
        assert cfg.num_subframes == 32
        assert cfg.device == "cpu"
    
    def test_sample(self):
        cfg = BlurConfig()
        params = cfg.sample()
        
        assert isinstance(params, BlurParams)
        assert cfg.shutter_length[0] <= params.shutter_length <= cfg.shutter_length[1]
        assert params.num_subframes == cfg.num_subframes
    
    def test_sample_deterministic(self):
        cfg = BlurConfig()
        rng = np.random.default_rng(42)
        
        params1 = cfg.sample(rng)
        rng = np.random.default_rng(42)
        params2 = cfg.sample(rng)
        
        assert params1.shutter_length == params2.shutter_length
        assert params1.camera_acceleration == params2.camera_acceleration
    
    def test_to_dict(self):
        cfg = BlurConfig()
        d = cfg.to_dict()
        
        assert "shutter_length" in d
        assert d["num_subframes"] == 32
    
    def test_from_dict(self):
        d = {"shutter_length": (0.3, 1.5), "num_subframes": 16}
        cfg = BlurConfig.from_dict(d)
        
        assert cfg.shutter_length == (0.3, 1.5)
        assert cfg.num_subframes == 16


class TestBlurParams:
    def test_time_grid_box(self):
        params = BlurParams(
            camera_translation_scale=1.0, camera_rotation_scale=0.0, camera_jerk=0.0,
            rolling_shutter_strength=0.0, object_scale=1.0, object_direction=0.0,
            object_nonrigid_noise=0.0, shutter_length=1.0, shutter_profile="box",
            shutter_skew=0.5, num_subframes=8, trajectory_profile="constant",
            camera_acceleration=0.0, lateral_acceleration=0.0, smooth_walk_jerk=0.0,
            occlusion_softness=0.0, noise_level=0.0, noise_poisson_scale=0.0,
            motion_sharpening=0.0, exposure_gain=1.0, depth_parallax_scale=1.0,
            se2_rotation_scale=1.0, object_model="se2", device="cpu"
        )
        
        t = params.get_time_grid()
        assert isinstance(t, torch.Tensor)
        assert len(t) == 8
        assert t[0].item() == pytest.approx(-1.0)
        assert t[-1].item() == pytest.approx(1.0)
    
    def test_time_grid_triangle(self):
        params = BlurParams(
            camera_translation_scale=1.0, camera_rotation_scale=0.0, camera_jerk=0.0,
            rolling_shutter_strength=0.0, object_scale=1.0, object_direction=0.0,
            object_nonrigid_noise=0.0, shutter_length=1.0, shutter_profile="triangle",
            shutter_skew=0.5, num_subframes=8, trajectory_profile="constant",
            camera_acceleration=0.0, lateral_acceleration=0.0, smooth_walk_jerk=0.0,
            occlusion_softness=0.0, noise_level=0.0, noise_poisson_scale=0.0,
            motion_sharpening=0.0, exposure_gain=1.0, depth_parallax_scale=1.0,
            se2_rotation_scale=1.0, object_model="se2", device="cpu"
        )
        
        result = params.get_time_grid()
        assert isinstance(result, tuple)
        t, w = result
        assert len(t) == 8
        assert len(w) == 8
        assert w.sum().item() == pytest.approx(1.0)
    
    def test_to_dict(self):
        cfg = BlurConfig()
        params = cfg.sample()
        d = params.to_dict()
        
        assert "shutter_length" in d
        assert "_camera_flow_fwd" not in d
    
    def test_from_dict(self):
        d = {"shutter_length": 1.5, "camera_acceleration": 2.0, "device": "cpu",
             "camera_translation_scale": 1.0, "camera_rotation_scale": 0.0,
             "camera_jerk": 0.0, "rolling_shutter_strength": 0.0, "object_scale": 1.0,
             "object_direction": 0.0, "object_nonrigid_noise": 0.0, "shutter_profile": "box",
             "shutter_skew": 0.5, "num_subframes": 32, "trajectory_profile": "constant",
             "lateral_acceleration": 0.0, "smooth_walk_jerk": 0.0, "occlusion_softness": 0.0,
             "noise_level": 0.0, "noise_poisson_scale": 0.0, "motion_sharpening": 0.0,
             "exposure_gain": 1.0, "depth_parallax_scale": 1.0, "se2_rotation_scale": 1.0,
             "object_model": "se2"}
        params = BlurParams.from_dict(d)
        
        assert params.shutter_length == 1.5
        assert params.camera_acceleration == 2.0
