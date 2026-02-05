"""Tests for BlurEngine."""

import pytest
import torch
import numpy as np

from blurgen.core import BlurConfig, BlurEngine


class TestBlurEngine:
    @pytest.fixture
    def engine(self):
        cfg = BlurConfig(device="cpu")
        return BlurEngine(cfg)
    
    @pytest.fixture
    def sample_inputs(self):
        H, W = 64, 64
        sharp_seq = torch.rand(3, 3, H, W)  # [T, C, H, W]
        depth = torch.ones(1, H, W)
        flow_fwd = torch.randn(2, H, W) * 2
        flow_bwd = -flow_fwd
        return sharp_seq, depth, flow_fwd, flow_bwd
    
    def test_synthesize_shape(self, engine, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        
        blur, meta = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd)
        
        assert blur.shape == (3, 64, 64)
        assert blur.min() >= 0.0
        assert blur.max() <= 1.0
    
    def test_synthesize_with_params(self, engine, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        params = engine.cfg.sample()
        
        blur, meta = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd, params=params)
        
        assert blur.shape == (3, 64, 64)
        assert meta["params"] is params
    
    def test_synthesize_batch(self, engine, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        
        # Add batch dimension
        sharp_seq = sharp_seq.unsqueeze(0)
        depth = depth.unsqueeze(0)
        flow_fwd = flow_fwd.unsqueeze(0)
        flow_bwd = flow_bwd.unsqueeze(0)
        
        blur, meta = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd)
        
        assert blur.shape == (1, 3, 64, 64)
    
    def test_meta_contains_trajectories(self, engine, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        
        blur, meta = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd)
        
        assert "traj" in meta
        assert "cam_traj" in meta
        assert "obj_traj" in meta
        assert "visibility" in meta
    
    def test_different_shutter_profiles(self, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        
        for profile in ["box", "triangle", "gaussian"]:
            cfg = BlurConfig(shutter_profile=profile, device="cpu")
            engine = BlurEngine(cfg)
            
            blur, _ = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd)
            assert blur.shape == (3, 64, 64)
    
    def test_different_camera_models(self, sample_inputs):
        sharp_seq, depth, flow_fwd, flow_bwd = sample_inputs
        
        for model in ["homography", "flow", "se2_rigid"]:
            cfg = BlurConfig(camera_model=model, device="cpu")
            engine = BlurEngine(cfg)
            
            blur, _ = engine.synthesize(sharp_seq, depth, flow_fwd, flow_bwd)
            assert blur.shape == (3, 64, 64)


class TestTrajectories:
    def test_camera_motion_output_shape(self):
        from blurgen.core import camera_motion, BlurConfig
        
        cfg = BlurConfig(device="cpu")
        params = cfg.sample()
        
        depth = torch.ones(1, 1, 32, 32)
        flow_fwd = torch.randn(1, 2, 32, 32)
        flow_bwd = -flow_fwd
        
        cam_traj = camera_motion(depth, flow_fwd, flow_bwd, params, params.num_subframes, flow_fwd, flow_bwd)
        
        assert cam_traj.shape == (1, params.num_subframes, 32, 32, 2)
    
    def test_object_motion_empty_masks(self):
        from blurgen.core import object_motion, BlurConfig
        
        cfg = BlurConfig(device="cpu")
        params = cfg.sample()
        
        flow_fwd = torch.randn(1, 2, 32, 32)
        flow_bwd = -flow_fwd
        
        obj_traj = object_motion(None, flow_fwd, flow_bwd, params, params.num_subframes)
        
        assert obj_traj.shape == (1, params.num_subframes, 32, 32, 2)
        assert (obj_traj == 0).all()
    
    def test_build_trajectory(self):
        from blurgen.core import build_trajectory
        
        cam = torch.randn(1, 16, 32, 32, 2)
        obj = torch.randn(1, 16, 32, 32, 2)
        
        traj = build_trajectory(cam, obj)
        
        assert traj.shape == (1, 16, 32, 32, 2)
        assert torch.allclose(traj, cam + obj)
