#!/usr/bin/env python3
"""Build conservative observed-free support counts from map-frame 3D rays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate low-height line-of-sight support above an explicit ground "
            "reference. The output is diagnostic evidence only and does not edit PGM."
        )
    )
    parser.add_argument("--rays", required=True, help="schema-v1 observation_rays.npz")
    parser.add_argument(
        "--ground-reference",
        required=True,
        help=(
            "geometry-only ground_reference.npy, typically produced by "
            "fit_ground_reference_plane.py; this is not semantic free evidence"
        ),
    )
    parser.add_argument("--grid-manifest", required=True, help="JSON containing grid metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-ground-relative-height-m", type=float, required=True)
    parser.add_argument("--max-ground-relative-height-m", type=float, required=True)
    parser.add_argument("--min-support-rays", type=int, required=True)
    parser.add_argument("--min-ray-range-m", type=float, default=0.0)
    parser.add_argument("--max-ray-range-m", type=float)
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

    from agt_map_reconstruction.maps.ground_aware_ray_evidence import (
        GroundAwareRayConfig,
        accumulate_ground_aware_ray_support,
    )
    from agt_map_reconstruction.maps.observation_ray_bundle import (
        load_observation_ray_bundle,
    )

    ray_path = Path(args.rays).expanduser().resolve()
    ground_path = Path(args.ground_reference).expanduser().resolve()
    manifest_path = Path(args.grid_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    grid_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_payload(grid_payload)
    bundle = load_observation_ray_bundle(ray_path, expected_frame_id=metadata.frame_id)
    ground_reference = np.load(ground_path, allow_pickle=False)
    config = GroundAwareRayConfig(
        min_ground_relative_height_m=args.min_ground_relative_height_m,
        max_ground_relative_height_m=args.max_ground_relative_height_m,
        min_support_rays=args.min_support_rays,
        min_ray_range_m=args.min_ray_range_m,
        max_ray_range_m=args.max_ray_range_m,
    )
    result = accumulate_ground_aware_ray_support(
        bundle,
        ground_reference,
        metadata,
        config,
    )

    np.save(output / "ray_free_support_count.npy", result["support_count"])
    np.save(
        output / "ray_free_support_mask.npy",
        result["support_mask"].astype(np.uint8),
    )
    manifest = {
        "schema_version": 1,
        "grid": metadata.to_dict(),
        "source_rays": str(ray_path),
        "source_ground_reference": str(ground_path),
        "source_grid_manifest": str(manifest_path),
        "ray_policy": {
            "min_ground_relative_height_m": float(config.min_ground_relative_height_m),
            "max_ground_relative_height_m": float(config.max_ground_relative_height_m),
            "min_support_rays": int(config.min_support_rays),
            "min_ray_range_m": float(config.min_ray_range_m),
            "max_ray_range_m": (
                None if config.max_ray_range_m is None else float(config.max_ray_range_m)
            ),
            "hit_cell_is_free": False,
            "requires_finite_ground_reference": True,
            "ground_reference_is_semantic_evidence": False,
            "semantic_promotion": False,
        },
        "summary": result["summary"],
    }
    (output / "observation_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("input_rays:", result["summary"]["input_ray_count"])
    print("accepted_rays:", result["summary"]["accepted_ray_count"])
    print("supported_cells:", result["summary"]["supported_cell_count"])
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
