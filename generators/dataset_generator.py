"""Dataset Generator: Parallel blur synthesis from CSV config."""

import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import traceback

from ..core import BlurConfig, BlurParams, BlurEngine
from ..codecs import TrajectoryCodec


@dataclass
class SampleSpec:
    """Specification for a single sample."""
    sample_id: str
    split: str
    sequence: str
    frame: str
    input_sharp: str
    input_flow_fwd: str
    input_flow_bwd: str
    output_blur: str
    output_flow: str
    output_traj: str
    output_sharp: str
    params: Dict[str, Any]


def _load_image(path: Path) -> np.ndarray:
    """Load image as float32 [H, W, 3] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _save_image(path: Path, img: np.ndarray) -> None:
    """Save float32 [H, W, 3] image in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _load_flow(path: Path) -> Optional[np.ndarray]:
    """Load flow [2, H, W] or [H, W, 2]."""
    if not path.exists():
        return None
    flow = np.load(path)
    if flow.ndim == 3 and flow.shape[-1] == 2:
        flow = flow.transpose(2, 0, 1)
    return flow.astype(np.float32)


def _process_sample(
    spec: SampleSpec,
    input_root: Path,
    output_root: Path,
    cfg_dict: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Process a single sample (worker function)."""
    try:
        # Build config and params
        cfg = BlurConfig.from_dict(cfg_dict)
        cfg.device = device
        
        params = BlurParams(
            camera_translation_scale=spec.params.get("camera_translation_scale", 1.0),
            camera_rotation_scale=spec.params.get("camera_rotation_scale", 0.0),
            camera_jerk=spec.params.get("camera_jerk", 0.0),
            rolling_shutter_strength=spec.params.get("rolling_shutter_strength", 0.0),
            object_scale=spec.params.get("object_scale", 1.0),
            object_direction=spec.params.get("object_direction", 0.0),
            object_nonrigid_noise=spec.params.get("object_nonrigid_noise", 0.0),
            shutter_length=spec.params.get("shutter_length", 1.0),
            shutter_profile=spec.params.get("shutter_profile", "box"),
            shutter_skew=spec.params.get("shutter_skew", 0.5),
            num_subframes=int(spec.params.get("num_subframes", 32)),
            trajectory_profile=spec.params.get("trajectory_profile", "acceleration"),
            camera_acceleration=spec.params.get("camera_acceleration", 0.0),
            lateral_acceleration=spec.params.get("lateral_acceleration", 0.0),
            smooth_walk_jerk=spec.params.get("smooth_walk_jerk", 0.0),
            occlusion_softness=spec.params.get("occlusion_softness", 0.0),
            noise_level=spec.params.get("noise_level", 0.0),
            noise_poisson_scale=spec.params.get("noise_poisson_scale", 0.0),
            motion_sharpening=spec.params.get("motion_sharpening", 0.0),
            exposure_gain=spec.params.get("exposure_gain", 1.0),
            depth_parallax_scale=spec.params.get("depth_parallax_scale", 1.0),
            se2_rotation_scale=spec.params.get("se2_rotation_scale", 1.0),
            object_model=spec.params.get("object_model", "se2"),
            device=device,
        )
        
        # Load inputs
        sharp_path = input_root / spec.input_sharp
        flow_fwd_path = input_root / spec.input_flow_fwd if spec.input_flow_fwd else None
        flow_bwd_path = input_root / spec.input_flow_bwd if spec.input_flow_bwd else None
        
        sharp = _load_image(sharp_path)  # [H, W, 3]
        H, W = sharp.shape[:2]
        
        # Load neighboring frames for SE2 extraction
        sharp_dir = sharp_path.parent
        frame_stem = sharp_path.stem
        
        # Try to find prev/next frames
        try:
            frame_num = int(frame_stem)
            prev_path = sharp_dir / f"{frame_num - 1:06d}.png"
            next_path = sharp_dir / f"{frame_num + 1:06d}.png"
            
            if prev_path.exists() and next_path.exists():
                sharp_prev = _load_image(prev_path)
                sharp_next = _load_image(next_path)
            else:
                sharp_prev = sharp
                sharp_next = sharp
        except ValueError:
            sharp_prev = sharp
            sharp_next = sharp
        
        # Create 3-frame sequence for SE2 extraction
        sharp_seq = np.stack([sharp_prev, sharp, sharp_next], axis=0)  # [3, H, W, 3]
        sharp_seq = torch.from_numpy(sharp_seq).permute(0, 3, 1, 2)  # [3, 3, H, W]
        
        # Load or create placeholder flows
        flow_fwd = _load_flow(flow_fwd_path) if flow_fwd_path else np.zeros((2, H, W), dtype=np.float32)
        flow_bwd = _load_flow(flow_bwd_path) if flow_bwd_path else np.zeros((2, H, W), dtype=np.float32)
        
        flow_fwd = torch.from_numpy(flow_fwd)
        flow_bwd = torch.from_numpy(flow_bwd)
        
        # Placeholder depth
        depth = torch.ones(1, H, W)
        
        # Move to device
        sharp_seq = sharp_seq.to(device)
        flow_fwd = flow_fwd.to(device)
        flow_bwd = flow_bwd.to(device)
        depth = depth.to(device)
        
        # Synthesize
        engine = BlurEngine(cfg)
        blur, meta = engine.synthesize(
            sharp_seq=sharp_seq,
            depth=depth,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            params=params,
        )
        
        # Move to CPU and convert
        blur_np = blur.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        
        # Save outputs
        output_blur_path = output_root / spec.output_blur
        output_traj_path = output_root / spec.output_traj
        output_sharp_path = output_root / spec.output_sharp
        
        _save_image(output_blur_path, blur_np)
        
        # Save sharp (copy or symlink)
        if not output_sharp_path.exists():
            output_sharp_path.parent.mkdir(parents=True, exist_ok=True)
            _save_image(output_sharp_path, sharp)
        
        # Save trajectories
        output_traj_path.parent.mkdir(parents=True, exist_ok=True)
        cam_traj = meta["cam_traj"].cpu().numpy().squeeze(0)  # [N, H, W, 2]
        obj_traj = meta["obj_traj"].cpu().numpy().squeeze(0) if meta["obj_traj"] is not None else None
        visibility = meta["visibility"].cpu().numpy().squeeze(0) if meta["visibility"] is not None else None
        
        TrajectoryCodec.save(
            output_traj_path,
            cam_traj=cam_traj,
            obj_traj=obj_traj,
            visibility=visibility,
            params=params.to_dict(),
            meta={"sample_id": spec.sample_id, "sequence": spec.sequence, "frame": spec.frame},
        )
        
        return {"sample_id": spec.sample_id, "status": "success"}
    
    except Exception as e:
        return {
            "sample_id": spec.sample_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


class DatasetGenerator:
    """Generate blur dataset from CSV configuration.
    
    Reads CSV with sample specs and generates blur images in parallel.
    """
    
    def __init__(
        self,
        csv_path: Path,
        input_root: Path,
        output_root: Path,
        config: Optional[BlurConfig] = None,
    ):
        """Initialize generator.
        
        Args:
            csv_path: path to CSV configuration
            input_root: root directory of input dataset
            output_root: root directory for output dataset
            config: BlurConfig (uses defaults if None)
        """
        self.csv_path = Path(csv_path)
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.config = config or BlurConfig()
        
        self.samples = self._load_csv()
    
    def _load_csv(self) -> List[SampleSpec]:
        """Load samples from CSV."""
        samples = []
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                params = {}
                for key in row:
                    if key not in ["sample_id", "split", "sequence", "frame",
                                   "input_sharp", "input_flow_fwd", "input_flow_bwd",
                                   "output_blur", "output_flow", "output_traj", "output_sharp"]:
                        try:
                            params[key] = float(row[key])
                        except (ValueError, TypeError):
                            params[key] = row[key]
                
                samples.append(SampleSpec(
                    sample_id=row["sample_id"],
                    split=row["split"],
                    sequence=row["sequence"],
                    frame=row["frame"],
                    input_sharp=row["input_sharp"],
                    input_flow_fwd=row.get("input_flow_fwd", ""),
                    input_flow_bwd=row.get("input_flow_bwd", ""),
                    output_blur=row["output_blur"],
                    output_flow=row.get("output_flow", ""),
                    output_traj=row["output_traj"],
                    output_sharp=row.get("output_sharp", ""),
                    params=params,
                ))
        return samples
    
    def generate(
        self,
        num_workers: int = 4,
        device: str = "cpu",
        skip_existing: bool = True,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Generate dataset in parallel.
        
        Args:
            num_workers: number of parallel workers
            device: "cpu" or "cuda"
            skip_existing: skip samples with existing output
            progress: show progress bar
        
        Returns:
            dict with generation statistics
        """
        # Filter samples
        samples_to_process = []
        for spec in self.samples:
            output_blur = self.output_root / spec.output_blur
            if skip_existing and output_blur.exists():
                continue
            samples_to_process.append(spec)
        
        if not samples_to_process:
            return {"total": len(self.samples), "processed": 0, "skipped": len(self.samples), "errors": []}
        
        cfg_dict = self.config.to_dict()
        
        results = {"total": len(self.samples), "processed": 0, "skipped": len(self.samples) - len(samples_to_process), "errors": []}
        
        if num_workers <= 1:
            # Sequential processing
            iterator = tqdm(samples_to_process, desc="Generating") if progress else samples_to_process
            for spec in iterator:
                result = _process_sample(spec, self.input_root, self.output_root, cfg_dict, device)
                if result["status"] == "success":
                    results["processed"] += 1
                else:
                    results["errors"].append(result)
        else:
            # Parallel processing (CPU only for multiprocessing)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_process_sample, spec, self.input_root, self.output_root, cfg_dict, "cpu"): spec
                    for spec in samples_to_process
                }
                
                iterator = tqdm(as_completed(futures), total=len(futures), desc="Generating") if progress else as_completed(futures)
                for future in iterator:
                    result = future.result()
                    if result["status"] == "success":
                        results["processed"] += 1
                    else:
                        results["errors"].append(result)
        
        return results
    
    def generate_single(self, sample_id: str, device: str = "cpu") -> Dict[str, Any]:
        """Generate a single sample by ID."""
        spec = next((s for s in self.samples if s.sample_id == sample_id), None)
        if spec is None:
            return {"status": "error", "error": f"Sample {sample_id} not found"}
        
        return _process_sample(spec, self.input_root, self.output_root, self.config.to_dict(), device)
