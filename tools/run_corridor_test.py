#!/usr/bin/env python3
"""Run reproducible EXP002 agricultural corridor recovery stages."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from agt_map_reconstruction.experiments.exp002 import (
    Exp002Config,
    build_run_id,
    create_run_directory,
    run_exp002_from_maps,
    sha256_file,
    write_exp002_results,
)
from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.grid_map import build_traversability_map


def _git_output(*args):
    process = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run EXP002 A/B/C agricultural corridor recovery.",
    )
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/EXP002"))
    parser.add_argument("--run-id", help="Immutable output directory name")
    parser.add_argument("--mode", choices=("all", "A", "B", "C"), default="all")
    parser.add_argument("--hash-pcd", action="store_true")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--row-height-threshold", type=float, default=0.20)
    parser.add_argument("--min-width-m", type=float, default=0.60)
    parser.add_argument("--max-width-m", type=float, default=2.00)
    parser.add_argument("--min-length-m", type=float, default=3.00)
    parser.add_argument("--min-row-coverage", type=float, default=0.70)
    parser.add_argument("--min-row-profile", type=float, default=0.25)
    parser.add_argument("--max-longitudinal-gap-m", type=float, default=0.05)
    parser.add_argument("--max-cross-row-gap-m", type=float, default=0.05)
    parser.add_argument("--max-boundary-gap-m", type=float, default=0.30)
    parser.add_argument("--direction-threshold", type=float, default=0.85)
    parser.add_argument("--min-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--baseline-min-cells", type=int, default=5)
    parser.add_argument("--component-min-cells", type=int, default=20)
    return parser.parse_args()


def main():
    args = _parse_args()
    pcd_path = args.pcd.resolve()
    commit = _git_output("rev-parse", "HEAD") or "nogit"
    run_id = args.run_id or build_run_id(commit)
    planned_run_dir = args.output / run_id
    if planned_run_dir.exists():
        raise FileExistsError(f"run directory already exists: {planned_run_dir}")
    config = Exp002Config(
        resolution=args.resolution,
        kernel_size=args.kernel_size,
        chunk_size=args.chunk_size,
        row_height_threshold=args.row_height_threshold,
        min_width_m=args.min_width_m,
        max_width_m=args.max_width_m,
        min_length_m=args.min_length_m,
        min_row_coverage=args.min_row_coverage,
        min_row_profile=args.min_row_profile,
        max_longitudinal_gap_m=args.max_longitudinal_gap_m,
        max_cross_row_gap_m=args.max_cross_row_gap_m,
        max_boundary_gap_m=args.max_boundary_gap_m,
        direction_threshold=args.direction_threshold,
        min_aspect_ratio=args.min_aspect_ratio,
        baseline_min_cells=args.baseline_min_cells,
        component_min_cells=args.component_min_cells,
    )
    points = load_pcd(pcd_path)
    maps = build_traversability_map(
        points,
        resolution=config.resolution,
        kernel_size=config.kernel_size,
        chunk_size=config.chunk_size,
    )
    result = run_exp002_from_maps(
        maps["height"],
        maps["relative_height"],
        maps["traversability"],
        config,
        origin_xy=maps["origin_xy"],
    )
    if args.mode != "all":
        result.stages = {args.mode: result.stages[args.mode]}

    run_dir = create_run_directory(args.output, run_id)
    metadata = {
        "experiment": "EXP002",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Aldoubt/Aldoubt-agt_map_reconstruction",
        "git_commit": commit,
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "mode": args.mode,
        "input_pcd": str(pcd_path),
        "input_size_bytes": pcd_path.stat().st_size,
        "input_points": int(len(points)),
    }
    if args.hash_pcd:
        metadata["input_sha256"] = sha256_file(pcd_path)

    write_exp002_results(result, run_dir, metadata)
    print(f"run_dir: {run_dir}")
    print(f"points: {len(points)}")
    print(f"row_angle_rad: {result.row_angle_rad}")
    for name, stage in result.stages.items():
        print(f"EXP002-{name} corridor_cells: {stage.metrics['corridor_cells']}")


if __name__ == "__main__":
    main()
