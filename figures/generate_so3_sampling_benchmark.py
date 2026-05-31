#!/usr/bin/env python3
"""
SO(3) sampler benchmark figure (panels 13–14 + legend via publication export).

Usage:
    python -m figures.generate_so3_sampling_benchmark              # pack NPZ
    python -m figures.generate_so3_sampling_benchmark --plot-only
    python -m figures.generate_so3_sampling_benchmark --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--refresh-grid", action="store_true")
    args = parser.parse_args()

    gen_args = [sys.executable, "-m", "generators.so3_sampling_benchmark"]
    if args.pack_only or args.plot_only:
        gen_args.append("--pack-only")
    elif args.force:
        gen_args.append("--refresh-all")
    else:
        if args.refresh_metrics:
            gen_args.append("--refresh-metrics")
        if args.refresh_grid:
            gen_args.append("--refresh-grid")
        if not (args.refresh_metrics or args.refresh_grid):
            gen_args.append("--pack-only")

    subprocess.check_call(gen_args)

    if not args.pack_only:
        subprocess.check_call(
            [sys.executable, "-m", "publication.export_figures", "--only", "13"]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
