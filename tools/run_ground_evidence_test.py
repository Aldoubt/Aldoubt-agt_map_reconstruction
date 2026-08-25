#!/usr/bin/env python3
"""Run reproducible EXP003 ground-evidence reconstruction."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from agt_map_reconstruction.experiments.exp003 import (
    Exp003Config,
    run_exp003,
    sha256_file,
    write_exp003_results,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_output(*args):
    process = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or f"exit status {process.returncode}"
        raise RuntimeError(f"Git command failed: {' '.join(args)}: {detail}")
    return process.stdout.strip()


def _git_provenance():
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def _pcd_identity(path):
    status = Path(path).stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
    )


def _snapshot_pcd(path, include_hash):
    path = Path(path)
    identity = _pcd_identity(path)
    digest = sha256_file(path) if include_hash else None
    if _pcd_identity(path) != identity:
        raise RuntimeError(f"input PCD changed while snapshotting: {path}")
    return {
        "identity": identity,
        "input_size_bytes": identity[2],
        "input_sha256": digest,
    }


def _verify_pcd_snapshot(path, snapshot):
    try:
        current_identity = _pcd_identity(path)
    except FileNotFoundError as error:
        raise RuntimeError(f"input PCD changed before publication: {path}") from error
    if current_identity != snapshot["identity"]:
        raise RuntimeError(f"input PCD changed before publication: {path}")


def _build_run_id(commit, instant=None):
    instant = instant or datetime.now(timezone.utc)
    instant = instant.astimezone(timezone.utc)
    return f"{instant.strftime('%Y%m%dT%H%M%SZ')}_{(commit or 'nogit')[:7]}"


def _run_id(value):
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError("run ID must be one non-empty path component")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run EXP003 conservative ground-evidence reconstruction.",
    )
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/EXP003"))
    parser.add_argument(
        "--run-id",
        type=_run_id,
        help="Immutable output directory name",
    )
    parser.add_argument("--hash-pcd", action="store_true")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--histogram-bins", type=int, default=64)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-ground-support-cells", type=int, default=2)
    parser.add_argument("--ground-window-m", type=float, default=0.50)
    parser.add_argument("--ground-percentile", type=float, default=20.0)
    parser.add_argument("--ground-seed-percentile", type=float, default=10.0)
    parser.add_argument("--max-ground-step-m", type=float, default=0.20)
    parser.add_argument("--max-interpolation-gap-m", type=float, default=0.25)
    parser.add_argument("--obstacle-height-m", type=float, default=0.15)
    parser.add_argument("--obstacle-inflation-radius-m", type=float, default=0.25)
    parser.add_argument("--interpolated-ground-cost", type=int, default=64)
    parser.add_argument(
        "--use-q90-for-obstacles",
        action="store_true",
        help="Use per-cell Q90 for obstacle evidence; off by default for greenhouse baseline",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    pcd_path = args.pcd.resolve()
    git_provenance = _git_provenance()
    commit = git_provenance["git_commit"]
    run_id = args.run_id or _build_run_id(commit)
    planned_run_dir = args.output / run_id
    if planned_run_dir.exists():
        raise FileExistsError(f"run directory already exists: {planned_run_dir}")

    config = Exp003Config(
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        low_quantile=args.low_quantile,
        histogram_bins=args.histogram_bins,
        min_points_per_cell=args.min_points_per_cell,
        min_ground_support_cells=args.min_ground_support_cells,
        ground_window_m=args.ground_window_m,
        ground_percentile=args.ground_percentile,
        ground_seed_percentile=args.ground_seed_percentile,
        max_ground_step_m=args.max_ground_step_m,
        max_interpolation_gap_m=args.max_interpolation_gap_m,
        obstacle_height_m=args.obstacle_height_m,
        obstacle_inflation_radius_m=args.obstacle_inflation_radius_m,
        interpolated_ground_cost=args.interpolated_ground_cost,
        use_q90_for_obstacles=args.use_q90_for_obstacles,
    )
    pcd_snapshot = _snapshot_pcd(pcd_path, args.hash_pcd)

    from agt_map_reconstruction.io.pcd_loader import load_pcd

    points = load_pcd(pcd_path)
    result = run_exp003(points, config)
    metadata = {
        "experiment": "EXP003",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Aldoubt/Aldoubt-agt_map_reconstruction",
        **git_provenance,
        "input_pcd": str(pcd_path),
        "input_size_bytes": pcd_snapshot["input_size_bytes"],
    }
    if pcd_snapshot["input_sha256"] is not None:
        metadata["input_sha256"] = pcd_snapshot["input_sha256"]
    _verify_pcd_snapshot(pcd_path, pcd_snapshot)
    write_exp003_results(result, planned_run_dir, metadata)
    print(f"run_dir: {planned_run_dir}")
    print(f"points: {result.input_points}")
    print(f"grid_shape_yx: {result.low_height.shape}")


if __name__ == "__main__":
    main()
