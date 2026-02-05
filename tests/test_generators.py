"""Tests for generators."""

import pytest
import tempfile
from pathlib import Path
import yaml
import csv

from blurgen.generators import CSVGenerator


class TestCSVGenerator:
    @pytest.fixture
    def sample_config(self):
        return {
            "dataset": {
                "name": "test_dataset",
                "root": "/tmp/test_data",
                "train_dir": "train",
                "test_dir": "test",
                "sharp_pattern": "*.png",
            },
            "output": {
                "root": "/tmp/output",
                "blur_subdir": "blur",
                "traj_subdir": "traj",
            },
            "params": {
                "shutter_length": {"distribution": "uniform", "min": 0.5, "max": 2.0},
                "camera_acceleration": {"distribution": "normal", "mean": 0.0, "std": 2.0, "min": -5.0, "max": 5.0},
                "shutter_profile": {"distribution": "fixed", "value": "box"},
            },
            "generation": {
                "seed": 42,
                "train_samples": 10,
                "test_samples": 5,
            },
        }
    
    def test_load_config(self, sample_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)
            
            gen = CSVGenerator(config_path)
            
            assert gen.dataset.name == "test_dataset"
            assert gen.seed == 42
    
    def test_param_specs(self, sample_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)
            
            gen = CSVGenerator(config_path)
            
            assert gen.param_specs["shutter_length"].distribution == "uniform"
            assert gen.param_specs["shutter_length"].min_val == 0.5
            assert gen.param_specs["camera_acceleration"].distribution == "normal"
            assert gen.param_specs["shutter_profile"].distribution == "fixed"
    
    def test_sampling_spec_uniform(self):
        from blurgen.generators.csv_generator import SamplingSpec
        import numpy as np
        
        spec = SamplingSpec(distribution="uniform", min_val=0.0, max_val=1.0)
        rng = np.random.default_rng(42)
        
        values = [spec.sample(rng) for _ in range(100)]
        
        assert all(0.0 <= v <= 1.0 for v in values)
    
    def test_sampling_spec_normal(self):
        from blurgen.generators.csv_generator import SamplingSpec
        import numpy as np
        
        spec = SamplingSpec(distribution="normal", mean=0.0, std=1.0, min_val=-3.0, max_val=3.0)
        rng = np.random.default_rng(42)
        
        values = [spec.sample(rng) for _ in range(100)]
        
        assert all(-3.0 <= v <= 3.0 for v in values)
        assert abs(np.mean(values)) < 0.5  # Should be around 0
    
    def test_sampling_spec_fixed(self):
        from blurgen.generators.csv_generator import SamplingSpec
        import numpy as np
        
        spec = SamplingSpec(distribution="fixed", value=42.0)
        rng = np.random.default_rng(42)
        
        values = [spec.sample(rng) for _ in range(10)]
        
        assert all(v == 42.0 for v in values)


class TestSamplingDeterminism:
    def test_same_seed_same_output(self):
        from blurgen.generators.csv_generator import SamplingSpec
        import numpy as np
        
        spec = SamplingSpec(distribution="uniform", min_val=0.0, max_val=1.0)
        
        rng1 = np.random.default_rng(42)
        values1 = [spec.sample(rng1) for _ in range(10)]
        
        rng2 = np.random.default_rng(42)
        values2 = [spec.sample(rng2) for _ in range(10)]
        
        assert values1 == values2
