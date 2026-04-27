#!/usr/bin/env python
"""Standalone visualization runner — loads a results CSV and renders figures."""

import argparse
from pathlib import Path
import pandas as pd

from config import Config
from visualization import render_all

def main():
    p = argparse.ArgumentParser(description="Render visualizations from a results CSV")
    p.add_argument("results_csv", type=Path, help="Path to results CSV file")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory for figures")
    args = p.parse_args()

    # Load results
    results = pd.read_csv(args.results_csv)
    print(f"Loaded {len(results)} rows from {args.results_csv}")

    # Create config with output dir
    cfg = Config()
    cfg.output_dir = args.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Render
    render_all(results, cfg)
    print(f"✓ Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
