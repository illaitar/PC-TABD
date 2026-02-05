"""CSV Configuration Generator: Generate sample configs from YAML spec."""

import yaml
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
import hashlib


@dataclass
class SamplingSpec:
    """Specification for parameter sampling."""
    distribution: str = "uniform"  # "uniform" | "normal" | "fixed"
    min_val: float = 0.0
    max_val: float = 1.0
    mean: float = 0.5
    std: float = 0.1
    value: Optional[float] = None  # for "fixed"
    
    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "fixed":
            return self.value if self.value is not None else self.mean
        elif self.distribution == "uniform":
            return float(rng.uniform(self.min_val, self.max_val))
        elif self.distribution == "normal":
            val = float(rng.normal(self.mean, self.std))
            return np.clip(val, self.min_val, self.max_val)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingSpec":
        return cls(
            distribution=d.get("distribution", "uniform"),
            min_val=d.get("min", 0.0),
            max_val=d.get("max", 1.0),
            mean=d.get("mean", (d.get("min", 0.0) + d.get("max", 1.0)) / 2),
            std=d.get("std", 0.1),
            value=d.get("value"),
        )


@dataclass
class DatasetSpec:
    """Specification for a dataset."""
    name: str
    root: str
    train_dir: str
    test_dir: str
    sharp_pattern: str = "*.png"  # glob pattern for sharp images
    flow_pattern: str = "*.npy"   # glob pattern for flow files


@dataclass
class OutputSpec:
    """Specification for output paths."""
    root: str
    blur_subdir: str = "blur"
    flow_subdir: str = "flow"
    traj_subdir: str = "trajectoriаes"
    sharp_subdir: str = "sharp"


class CSVGenerator:
    """Generate CSV configuration files for dataset generation.
    
    YAML config format:
    ```yaml
    dataset:
      name: gopro
      root: /path/to/gopro
      train_dir: train
      test_dir: test
      sharp_pattern: "*.png"
      flow_pattern: "*.npy"
    
    output:
      root: /path/to/output
      blur_subdir: blur
      traj_subdir: trajectories
    
    params:
      shutter_length:
        distribution: uniform
        min: 0.5
        max: 2.0
      camera_acceleration:
        distribution: normal
        mean: 0.0
        std: 2.0
        min: -5.0
        max: 5.0
      shutter_profile:
        distribution: fixed
        value: box
    
    generation:
      seed: 42
      train_samples: 1000
      test_samples: 100
    ```
    """
    
    PARAM_COLUMNS = [
        "shutter_length",
        "shutter_profile",
        "shutter_skew",
        "camera_translation_scale",
        "camera_rotation_scale",
        "camera_jerk",
        "camera_acceleration",
        "lateral_acceleration",
        "rolling_shutter_strength",
        "object_scale",
        "object_direction",
        "object_nonrigid_noise",
        "trajectory_profile",
        "smooth_walk_jerk",
        "noise_level",
        "noise_poisson_scale",
        "motion_sharpening",
        "exposure_gain",
        "depth_parallax_scale",
        "se2_rotation_scale",
        "occlusion_softness",
        "object_model",
        "bg_fill",
        "num_subframes",
    ]
    
    DEFAULT_SPECS = {
        "shutter_length": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "shutter_profile": {"distribution": "fixed", "value": "box"},
        "shutter_skew": {"distribution": "uniform", "min": 0.3, "max": 0.7},
        "camera_translation_scale": {"distribution": "uniform", "min": 0.3, "max": 1.5},
        "camera_rotation_scale": {"distribution": "uniform", "min": 0.0, "max": 0.5},
        "camera_jerk": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        "camera_acceleration": {"distribution": "uniform", "min": -4.5, "max": 4.5},
        "lateral_acceleration": {"distribution": "uniform", "min": -1.7, "max": 1.7},
        "rolling_shutter_strength": {"distribution": "uniform", "min": 0.0, "max": 0.3},
        "object_scale": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "object_direction": {"distribution": "fixed", "value": 0.0},
        "object_nonrigid_noise": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "trajectory_profile": {"distribution": "fixed", "value": "acceleration"},
        "smooth_walk_jerk": {"distribution": "uniform", "min": 0.0, "max": 0.02},
        "noise_level": {"distribution": "uniform", "min": 0.0, "max": 0.02},
        "noise_poisson_scale": {"distribution": "uniform", "min": 0.0, "max": 0.01},
        "motion_sharpening": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "exposure_gain": {"distribution": "uniform", "min": 0.9, "max": 1.1},
        "depth_parallax_scale": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "se2_rotation_scale": {"distribution": "uniform", "min": 1.0, "max": 1.0},
        "occlusion_softness": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "object_model": {"distribution": "fixed", "value": "se2"},
        "bg_fill": {"distribution": "fixed", "value": "inpaint"},
        "num_subframes": {"distribution": "fixed", "value": 32},
    }
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize generator from YAML config."""
        self.config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.dataset = DatasetSpec(
            name=self.config["dataset"]["name"],
            root=self.config["dataset"]["root"],
            train_dir=self.config["dataset"].get("train_dir", "train"),
            test_dir=self.config["dataset"].get("test_dir", "test"),
            sharp_pattern=self.config["dataset"].get("sharp_pattern", "*.png"),
            flow_pattern=self.config["dataset"].get("flow_pattern", "*.npy"),
        )
        
        self.output = OutputSpec(
            root=self.config["output"]["root"],
            blur_subdir=self.config["output"].get("blur_subdir", "blur"),
            flow_subdir=self.config["output"].get("flow_subdir", "flow"),
            traj_subdir=self.config["output"].get("traj_subdir", "trajectories"),
            sharp_subdir=self.config["output"].get("sharp_subdir", "sharp"),
        )
        
        self.param_specs = {}
        for param in self.PARAM_COLUMNS:
            if "params" in self.config and param in self.config["params"]:
                self.param_specs[param] = SamplingSpec.from_dict(self.config["params"][param])
            else:
                self.param_specs[param] = SamplingSpec.from_dict(self.DEFAULT_SPECS.get(param, {}))
        
        gen_cfg = self.config.get("generation", {})
        self.seed = gen_cfg.get("seed", 42)
        self.train_samples = gen_cfg.get("train_samples", None)
        self.test_samples = gen_cfg.get("test_samples", None)
    
    def _discover_samples(self, split: str) -> List[Dict[str, str]]:
        """Discover samples in dataset split."""
        split_dir = self.dataset.train_dir if split == "train" else self.dataset.test_dir
        root = Path(self.dataset.root) / split_dir
        
        samples = []
        
        # Assume structure: root/sequence/sharp/*.png, root/sequence/flow/*.npy
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            sharp_dir = seq_dir / "sharp"
            flow_dir = seq_dir
            
            if not sharp_dir.exists():
                sharp_dir = seq_dir  # Flat structure
            if not flow_dir.exists():
                flow_dir = seq_dir
            
            sharp_files = sorted(sharp_dir.glob(self.dataset.sharp_pattern))
            
            for sharp_file in sharp_files:
                stem = sharp_file.stem
                flow_fwd = flow_dir / "flow_cn_pix" / f"{stem}_fwd.npy"
                flow_bwd = flow_dir / "flow_cp_pix" / f"{stem}_bwd.npy"
                
                # Alternative naming
                if not flow_fwd.exists():
                    flow_fwd = flow_dir / f"flow_fwd_{stem}.npy"
                if not flow_bwd.exists():
                    flow_bwd = flow_dir / f"flow_bwd_{stem}.npy"
                
                samples.append({
                    "sequence": seq_dir.name,
                    "frame": stem,
                    "sharp_path": str(sharp_file.relative_to(self.dataset.root)),
                    "flow_fwd_path": str(flow_fwd.relative_to(self.dataset.root)) if flow_fwd.exists() else "",
                    "flow_bwd_path": str(flow_bwd.relative_to(self.dataset.root)) if flow_bwd.exists() else "",
                })
        
        return samples
    
    def _generate_output_paths(self, sample: Dict[str, str], sample_id: str) -> Dict[str, str]:
        """Generate output paths for a sample."""
        seq = sample["sequence"]
        frame = sample["frame"]
        
        return {
            "blur_path": f"{self.output.blur_subdir}/{seq}/{frame}.png",
            "flow_path": f"{self.output.flow_subdir}/{seq}/{frame}.npy",
            "traj_path": f"{self.output.traj_subdir}/{seq}/{frame}.npy",
            "sharp_path": f"{self.output.sharp_subdir}/{seq}/{frame}.png",
            "sample_id": sample_id,
        }
    
    def generate(self, output_csv: Union[str, Path], split: str = "train") -> int:
        """Generate CSV configuration file.
        
        Args:
            output_csv: path to output CSV
            split: "train" or "test"
        
        Returns:
            number of samples generated
        """
        rng = np.random.default_rng(self.seed if split == "train" else self.seed + 1)
        
        samples = self._discover_samples(split)
        max_samples = self.train_samples if split == "train" else self.test_samples
        
        if max_samples is not None and len(samples) > max_samples:
            indices = rng.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in sorted(indices)]
        
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        columns = [
            "sample_id", "split", "sequence", "frame",
            "input_sharp", "input_flow_fwd", "input_flow_bwd",
            "output_blur", "output_flow", "output_traj", "output_sharp",
        ] + self.PARAM_COLUMNS
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for i, sample in enumerate(samples):
                sample_id = hashlib.md5(f"{split}_{sample['sequence']}_{sample['frame']}".encode()).hexdigest()[:12]
                output_paths = self._generate_output_paths(sample, sample_id)
                
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "sequence": sample["sequence"],
                    "frame": sample["frame"],
                    "input_sharp": sample["sharp_path"],
                    "input_flow_fwd": sample["flow_fwd_path"],
                    "input_flow_bwd": sample["flow_bwd_path"],
                    "output_blur": output_paths["blur_path"],
                    "output_flow": output_paths["flow_path"],
                    "output_traj": output_paths["traj_path"],
                    "output_sharp": output_paths["sharp_path"],
                }
                
                for param in self.PARAM_COLUMNS:
                    spec = self.param_specs[param]
                    if spec.distribution == "fixed":
                        row[param] = spec.value if spec.value is not None else spec.mean
                    else:
                        row[param] = spec.sample(rng)
                
                writer.writerow(row)
        
        return len(samples)
    
    def generate_all(self, output_dir: Union[str, Path]) -> Tuple[int, int]:
        """Generate CSV files for both train and test splits.
        
        Args:
            output_dir: directory to save CSV files
        
        Returns:
            (train_count, test_count)
        """
        output_dir = Path(output_dir)
        train_csv = output_dir / f"{self.dataset.name}_train.csv"
        test_csv = output_dir / f"{self.dataset.name}_test.csv"
        
        train_count = self.generate(train_csv, "train")
        test_count = self.generate(test_csv, "test")
        
        return train_count, test_count
