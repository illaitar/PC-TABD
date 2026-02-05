"""CLI for generating CSV configuration files."""

import argparse
from pathlib import Path

from blurgen.generators import CSVGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate CSV configuration for blur dataset")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for CSV files")
    parser.add_argument("--train-only", action="store_true", help="Generate only train CSV")
    parser.add_argument("--test-only", action="store_true", help="Generate only test CSV")
    
    args = parser.parse_args()
    
    gen = CSVGenerator(args.config)
    
    if args.train_only:
        train_csv = args.output / f"{gen.dataset.name}_train.csv"
        n = gen.generate(train_csv, "train")
        print(f"Generated {n} train samples -> {train_csv}")
    elif args.test_only:
        test_csv = args.output / f"{gen.dataset.name}_test.csv"
        n = gen.generate(test_csv, "test")
        print(f"Generated {n} test samples -> {test_csv}")
    else:
        train_n, test_n = gen.generate_all(args.output)
        print(f"Generated {train_n} train + {test_n} test samples -> {args.output}")


if __name__ == "__main__":
    main()
