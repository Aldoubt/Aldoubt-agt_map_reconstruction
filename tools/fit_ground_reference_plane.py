#!/usr/bin/env python3
"""Fit a geometry-only affine ground reference for P1-E ray evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit an affine z(x,y) reference from finite ground_surface cells. "
            "The extrapolated result is not semantic free-space evidence."
        )
    )
    parser.add_argument("--ground-surface", required=True)
    parser.add_argument("--grid-manifest", required=True)
    parser.add_argument("--output", required=True)
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


def main():
    args = build_parser().parse_args()
    from agt_map_reconstruction.maps.ground_reference_plane import (
        fit_affine_ground_reference,
    )

    ground_path = Path(args.ground_surface).expanduser().resolve()
    manifest_path = Path(args.grid_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_payload(payload)
    ground_surface = np.load(ground_path, allow_pickle=False)
    result = fit_affine_ground_reference(ground_surface, metadata)

    np.save(output / "ground_reference.npy", result["ground_reference"])
    np.save(
        output / "ground_reference_source_mask.npy",
        result["finite_support_mask"].astype(np.uint8),
    )
    model = dict(result["model"])
    model.update({
        "grid": metadata.to_dict(),
        "source_ground_surface": str(ground_path),
        "source_grid_manifest": str(manifest_path),
        "interpretation": (
            "geometry-only z reference for 3D ray height; extrapolation is not "
            "observed-free evidence and does not change semantic labels"
        ),
    })
    (output / "ground_reference_manifest.json").write_text(
        json.dumps(model, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("support_cells:", model["support_cell_count"])
    print("extrapolated_cells:", model["extrapolated_cell_count"])
    print(f"residual_rmse_m: {model['residual_rmse_m']:.6f}")
    print(f"residual_p95_abs_m: {model['residual_p95_abs_m']:.6f}")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
