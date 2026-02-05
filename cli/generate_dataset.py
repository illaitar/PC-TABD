"""CLI for generating blur datasets from CSV configuration."""

import argparse
from pathlib import Path

from blurgen import BlurConfig
from blurgen.generators import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate blur dataset from CSV configuration")
    parser.add_argument("csv", type=Path, help="Path to CSV configuration file")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input dataset root directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output dataset root directory")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing outputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    parser.add_argument("--camera-model", type=str, default="se2_rigid", help="Camera model")
    parser.add_argument("--object-threshold", type=float, default=3.0, help="Object residual threshold")
    parser.add_argument("--object-saturation", type=float, default=10.0, help="Object saturation")
    
    args = parser.parse_args()
    
    cfg = BlurConfig(
        device=args.device,
        camera_model=args.camera_model,
        object_residual_threshold=args.object_threshold,
        object_saturation=args.object_saturation,
        object_soft_blend=True,
    )
    gen = DatasetGenerator(args.csv, args.input, args.output, cfg)
    
    results = gen.generate(
        num_workers=args.workers,
        device=args.device,
        skip_existing=not args.no_skip,
        progress=not args.no_progress,
    )
    
    print(f"\nGeneration complete:")
    print(f"  Total samples: {results['total']}")
    print(f"  Processed: {results['processed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\nErrors:")
        for err in results['errors'][:10]:
            print(f"  {err['sample_id']}: {err['error']}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")


if __name__ == "__main__":
    main()
