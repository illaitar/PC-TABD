"""CLI for generating blur datasets using pc-tabd engine."""

import argparse
import sys
from pathlib import Path

import csv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from blurgen.pctabd import SIN3DConfig, SIN3DEngine


def load_image(path: Path) -> torch.Tensor:
    """Load image as [3, H, W] tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(path: Path, tensor: torch.Tensor):
    """Save [3, H, W] tensor to image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        arr = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        arr = tensor.cpu().numpy()
    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_flow(path: Path) -> torch.Tensor:
    """Load flow as [2, H, W] tensor."""
    flow = np.load(path).astype(np.float32)
    if flow.ndim == 3 and flow.shape[-1] == 2:
        flow = flow.transpose(2, 0, 1)
    return torch.from_numpy(flow)


def discover_gopro_samples(root: Path, split: str = "train"):
    """Discover GoPro samples with flow data and neighboring frames."""
    split_dir = root / split
    samples = []
    
    for seq_dir in sorted(split_dir.iterdir()):
        if not seq_dir.is_dir() or seq_dir.name.startswith("."):
            continue
        
        sharp_dir = seq_dir / "sharp"
        flow_fwd_dir = seq_dir / "flow_cn_pix"
        flow_bwd_dir = seq_dir / "flow_cp_pix"
        
        if not sharp_dir.exists() or not flow_fwd_dir.exists():
            continue
        
        frames = sorted([f.stem for f in sharp_dir.glob("*.png")])
        
        for i, stem in enumerate(frames):
            if i == 0 or i == len(frames) - 1:
                continue
            
            flow_fwd = flow_fwd_dir / f"{stem}.npy"
            flow_bwd = flow_bwd_dir / f"{stem}.npy"
            
            if not flow_fwd.exists() or not flow_bwd.exists():
                continue
            
            prev_stem = frames[i - 1]
            next_stem = frames[i + 1]
            
            samples.append({
                "sequence": seq_dir.name,
                "frame": stem,
                "sharp_path": sharp_dir / f"{stem}.png",
                "sharp_prev_path": sharp_dir / f"{prev_stem}.png",
                "sharp_next_path": sharp_dir / f"{next_stem}.png",
                "flow_fwd_path": flow_fwd,
                "flow_bwd_path": flow_bwd,
            })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate blur dataset using pc-tabd")
    parser.add_argument("-i", "--input", type=Path, required=True, help="GoPro dataset root")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("-n", "--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Discovering samples in {args.input}...")
    all_samples = discover_gopro_samples(args.input, "train")
    print(f"Found {len(all_samples)} total frames")
    
    # Random sample
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_samples), min(args.num_samples, len(all_samples)), replace=False)
    samples = [all_samples[i] for i in indices]
    print(f"Selected {len(samples)} samples")
    
    # Config - stronger blur, CPU only
    cfg = SIN3DConfig(
        device="cpu",
        num_subframes=32,
        camera_model="se2_rigid",
        trajectory_profile="acceleration",
        camera_translation_scale=(0.8, 1.5),
        camera_acceleration=(1.5, 4.0),
        lateral_acceleration=(-0.3, 0.3),
        camera_jerk=(0.0, 0.02),
        shutter_length=(1.2, 2.0),  # longer shutter = more blur
        rolling_shutter_strength=(0.0, 0.15),
        exposure_gain=(0.98, 1.02),
        noise_level=(0.0, 0.0),
        noise_poisson_scale=(0.0, 0.0),
        motion_sharpening=(0.0, 0.0),
        depth_parallax_scale=(0.8, 1.2),
        object_nonrigid_noise=(0.0, 0.0),
        object_scale_range=(1.0, 1.0),
    )
    
    engine = SIN3DEngine(
        cfg,
        use_linear_light=True,
        use_depth_ordering=False,  # no visibility computation
        extract_objects=False,  # no object motion
    )
    
    # Generate
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "samples.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "sequence", "frame", "shutter_length",
            "camera_acceleration", "lateral_acceleration", "object_scale",
            "sharp_path", "blur_path", "traj_path", "flow_fwd_path", "flow_bwd_path",
        ])
        writer.writeheader()
        
        for idx, sample in enumerate(tqdm(samples, desc="Generating")):
            try:
                seq_name = sample["sequence"]
                frame_name = sample["frame"]
                
                # Load data
                sharp = load_image(sample["sharp_path"])
                sharp_prev = load_image(sample["sharp_prev_path"])
                sharp_next = load_image(sample["sharp_next_path"])
                flow_fwd = load_flow(sample["flow_fwd_path"])
                flow_bwd = load_flow(sample["flow_bwd_path"])
                
                # Create 3-frame sequence
                sharp_seq = torch.stack([sharp_prev, sharp, sharp_next], dim=0)
                
                H, W = sharp.shape[1], sharp.shape[2]
                depth = torch.zeros(1, 1, H, W)
                
                # Keep on CPU
                flow_fwd_dev = flow_fwd
                flow_bwd_dev = flow_bwd
                
                # Sample params
                params = cfg.sample()
                
                # Synthesize
                with torch.no_grad():
                    blur, meta = engine.synthesize(sharp_seq, depth, flow_fwd_dev, flow_bwd_dev, params=params)
                
                # Output paths
                blur_path = args.output / "blur" / seq_name / f"{frame_name}.png"
                sharp_out_path = args.output / "sharp" / seq_name / f"{frame_name}.png"
                traj_path = args.output / "trajectories" / seq_name / f"{frame_name}.npy"
                flow_fwd_out = args.output / "flow_fwd" / seq_name / f"{frame_name}.npy"
                flow_bwd_out = args.output / "flow_bwd" / seq_name / f"{frame_name}.npy"
                
                # Save blur
                save_image(blur_path, blur.squeeze(0) if blur.dim() == 4 else blur)
                
                # Save sharp (once per frame)
                if not sharp_out_path.exists():
                    save_image(sharp_out_path, sharp)
                
                # Save trajectories
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                cam_traj = meta["cam_traj"].cpu().numpy()
                if cam_traj.ndim == 5:
                    cam_traj = cam_traj[0]  # [N, H, W, 2]
                np.save(traj_path, cam_traj.astype(np.float16))
                
                # Save flows (once per frame)
                if not flow_fwd_out.exists():
                    flow_fwd_out.parent.mkdir(parents=True, exist_ok=True)
                    flow_bwd_out.parent.mkdir(parents=True, exist_ok=True)
                    flow_fwd_np = flow_fwd.cpu().numpy()
                    flow_bwd_np = flow_bwd.cpu().numpy()
                    if flow_fwd_np.shape[0] == 2:
                        flow_fwd_np = flow_fwd_np.transpose(1, 2, 0)  # [H, W, 2]
                        flow_bwd_np = flow_bwd_np.transpose(1, 2, 0)
                    np.save(flow_fwd_out, flow_fwd_np.astype(np.float16))
                    np.save(flow_bwd_out, flow_bwd_np.astype(np.float16))
                
                writer.writerow({
                    "idx": idx,
                    "sequence": seq_name,
                    "frame": frame_name,
                    "shutter_length": params.shutter_length,
                    "camera_acceleration": params.camera_acceleration,
                    "lateral_acceleration": params.lateral_acceleration,
                    "object_scale": params.object_scale,
                    "sharp_path": f"sharp/{seq_name}/{frame_name}.png",
                    "blur_path": f"blur/{seq_name}/{frame_name}.png",
                    "traj_path": f"trajectories/{seq_name}/{frame_name}.npy",
                    "flow_fwd_path": f"flow_fwd/{seq_name}/{frame_name}.npy",
                    "flow_bwd_path": f"flow_bwd/{seq_name}/{frame_name}.npy",
                })
                
            except Exception as e:
                print(f"Error processing {sample['sequence']}/{sample['frame']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nDone! Output: {args.output}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
