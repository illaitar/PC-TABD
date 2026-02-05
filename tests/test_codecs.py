"""Tests for TrajectoryCodec."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from blurgen.codecs import TrajectoryCodec


class TestTrajectoryCodec:
    @pytest.fixture
    def sample_data(self):
        N, H, W = 16, 32, 32
        return {
            "cam_traj": np.random.randn(N, H, W, 2).astype(np.float32),
            "obj_traj": np.random.randn(N, H, W, 2).astype(np.float32),
            "visibility": np.random.rand(N, H, W).astype(np.float32),
            "params": {"shutter_length": 1.5, "camera_acceleration": 2.0},
            "meta": {"sample_id": "test123"},
        }
    
    def test_encode_decode(self, sample_data):
        # Use compress=False for exact round-trip
        encoded = TrajectoryCodec.encode(compress=False, **sample_data)
        decoded = TrajectoryCodec.decode(encoded)
        
        assert np.allclose(decoded["cam_traj"], sample_data["cam_traj"], atol=1e-5)
        assert np.allclose(decoded["traj"], sample_data["cam_traj"] + sample_data["obj_traj"], atol=1e-5)
        assert decoded["params"]["shutter_length"] == 1.5
    
    def test_encode_minimal(self):
        cam_traj = np.random.randn(8, 16, 16, 2).astype(np.float32)
        
        encoded = TrajectoryCodec.encode(cam_traj=cam_traj)
        decoded = TrajectoryCodec.decode(encoded)
        
        assert np.allclose(decoded["cam_traj"], cam_traj, atol=1e-3)
        assert np.allclose(decoded["traj"], cam_traj, atol=1e-3)
    
    def test_save_load(self, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traj.npy"
            
            TrajectoryCodec.save(path, compress=False, **sample_data)
            loaded = TrajectoryCodec.load(path)
            
            assert np.allclose(loaded["cam_traj"], sample_data["cam_traj"], atol=1e-5)
            assert loaded["meta"]["sample_id"] == "test123"
    
    def test_compression(self, sample_data):
        # With compression (default)
        encoded_compressed = TrajectoryCodec.encode(compress=True, **sample_data)
        
        # Without compression
        encoded_full = TrajectoryCodec.encode(compress=False, **sample_data)
        
        assert encoded_compressed["cam_traj"].dtype == np.float16
        assert encoded_full["cam_traj"].dtype == np.float32
    
    def test_version(self, sample_data):
        encoded = TrajectoryCodec.encode(**sample_data)
        assert encoded["version"] == TrajectoryCodec.VERSION
