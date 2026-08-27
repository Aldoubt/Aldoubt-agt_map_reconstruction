#!/usr/bin/env python3
"""Fit a KNN local affine ground-height reference and report confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a local affine ground-height reference from finite PCD-derived "
            "ground support. This is geometry-only and never promotes semantic free space."
        )
    )
    parser.add_argument("--ground-surface", required=True)
    parser.add_argument("--grid-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbors", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    return parser


def _metadata_from_payload(payload):
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("grid manifest must contain a grid mapping")
    origin = grid.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("grid.origin must contain [x, y, yaw]")
    return GridMetadata(
        resolution=float(grid["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=int(grid["width"]),
        height=int(grid["height"]),
        frame_id=str(grid.get("frame_id", "map")),
    )


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.local_ground_reference import (
        fit_knn_local_affine_ground_reference,
    )

    ground_path = Path(args.ground_surface).expanduser().resolve()
    manifest_path = Path(args.grid_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_payload(payload)
    ground = np.load(ground_path, allow_pickle=False)

    result = fit_knn_local_affine_ground_reference(
        ground,
        metadata,
        neighbor_count=args.neighbors,
        chunk_size=args.chunk_size,
    )

    np.save(output / "ground_reference.npy", result["ground_reference"])
    np.save(
        output / "ground_reference_nearest_support_distance.npy",
        result["nearest_support_distance_m"],
    )
    np.save(
        output / "ground_reference_valid_mask.npy",
        result["valid_fit_mask"].astype(np.uint8),
    )
    manifest = {
        "schema_version": 1,
        "grid": metadata.to_dict(),
        "source_ground_surface": str(ground_path),
        "source_grid_manifest": str(manifest_path),
        "model": result["model"],
        "global_affine_baseline": result["global_affine_baseline"],
        "policy": {
            "ground_reference_is_semantic_evidence": False,
            "semantic_promotion": False,
        },
    }
    (output / "ground_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    local = result["model"]
    global_ = result["global_affine_baseline"]
    print("output:", output)
    print("neighbors:", local["neighbor_count"])
    print("support_cells:", local["support_cell_count"])
    print("unknown_cells:", local["unknown_cell_count"])
    print("invalid_fit_cells:", local["invalid_fit_cell_count"])
    print("local_cv_rmse_m:", _fmt(local["cv_residual_rmse_m"]))
    print("local_cv_p95_abs_m:", _fmt(local["cv_residual_p95_abs_m"]))
    print("global_affine_rmse_m:", _fmt(global_["residual_rmse_m"]))
    print("global_affine_p95_abs_m:", _fmt(global_["residual_p95_abs_m"]))
    print(
        "unknown_support_distance_median_m:",
        _fmt(local["unknown_nearest_support_distance_median_m"]),
    )
    print(
        "unknown_support_distance_p95_m:",
        _fmt(local["unknown_nearest_support_distance_p95_m"]),
    )
    print(
        "unknown_support_distance_max_m:",
        _fmt(local["unknown_nearest_support_distance_max_m"]),
    )
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
