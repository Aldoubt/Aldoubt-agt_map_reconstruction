#!/usr/bin/env python3
"""Evaluate adjacent-aisle connectivity through finite headland domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen baseline and conservative maps on side-local, "
            "adjacent-aisle headland handoff connectivity."
        )
    )
    parser.add_argument("--baseline-map", required=True)
    parser.add_argument("--conservative-map", required=True)
    parser.add_argument("--aisles", required=True)
    parser.add_argument("--depth-profile", required=True)
    parser.add_argument("--baseline-handoffs")
    parser.add_argument("--conservative-handoffs")
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read map: {path}")
    return path, np.flipud(image).astype(np.uint8, copy=False)


def _load_aisles(path):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rectangles = payload.get("rectangles") if isinstance(payload, dict) else None
    if not isinstance(rectangles, list):
        raise ValueError("aisle JSON must contain rectangles")
    grid = dict(payload.get("grid") or {})
    if "resolution" not in grid:
        raise ValueError("aisle JSON grid.resolution is required")
    return path, payload, rectangles, grid


def _load_depth_profile(path):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("depth profile must contain mask_files")
    masks = {
        key: np.load(path.parent / filename, allow_pickle=False).astype(bool, copy=False)
        for key, filename in files.items()
    }
    return path, payload, masks


def _load_handoffs(path):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("handoffs") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"handoff JSON must contain a handoffs list: {path}")
    return path, records


def _metadata_from_grid(grid):
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    origin = grid.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("aisle grid.origin must contain [x, y, yaw]")
    return GridMetadata(
        resolution=float(grid["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=int(grid["width"]),
        height=int(grid["height"]),
        frame_id=str(grid.get("frame_id", "map")),
    )


def _compute_handoffs(base_map, aisles, grid, radius):
    from agt_map_reconstruction.maps.aisle_handoff_boundary import (
        estimate_aisle_handoff_boundary,
    )

    metadata = _metadata_from_grid(grid)
    return [
        estimate_aisle_handoff_boundary(
            base_map,
            aisle,
            resolution=float(metadata.resolution),
            radius_m=float(radius),
            metadata=metadata,
        )
        for aisle in aisles
    ]


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_handoff_connectivity import (
        analyze_headland_handoff_connectivity,
        write_headland_connectivity_bundle,
    )

    baseline_path, baseline = _read_pgm(args.baseline_map)
    conservative_path, conservative = _read_pgm(args.conservative_map)
    aisle_path, _, aisles, grid = _load_aisles(args.aisles)
    profile_path, profile, masks = _load_depth_profile(args.depth_profile)

    if baseline.shape != conservative.shape:
        raise ValueError("baseline and conservative maps must have the same shape")
    expected_shape = (
        int(grid.get("height", baseline.shape[0])),
        int(grid.get("width", baseline.shape[1])),
    )
    if baseline.shape != expected_shape:
        raise ValueError(
            f"map shape {baseline.shape} does not match aisle grid {expected_shape}"
        )

    if bool(args.baseline_handoffs) != bool(args.conservative_handoffs):
        raise ValueError(
            "--baseline-handoffs and --conservative-handoffs must be supplied together"
        )

    if args.baseline_handoffs:
        baseline_handoff_path, baseline_handoffs = _load_handoffs(
            args.baseline_handoffs
        )
        conservative_handoff_path, conservative_handoffs = _load_handoffs(
            args.conservative_handoffs
        )
        handoff_source = "provided_frozen_json"
    else:
        baseline_handoff_path = None
        conservative_handoff_path = None
        baseline_handoffs = _compute_handoffs(
            baseline, aisles, grid, args.radius
        )
        conservative_handoffs = _compute_handoffs(
            conservative, aisles, grid, args.radius
        )
        handoff_source = "recomputed_from_each_map"

    result = analyze_headland_handoff_connectivity(
        baseline,
        conservative,
        aisles,
        baseline_handoffs,
        conservative_handoffs,
        profile,
        masks,
        resolution=float(grid["resolution"]),
        radius_m=float(args.radius),
    )
    result["sources"] = {
        "baseline_map": str(baseline_path),
        "conservative_map": str(conservative_path),
        "aisles": str(aisle_path),
        "depth_profile": str(profile_path),
        "baseline_handoffs": (
            None if baseline_handoff_path is None else str(baseline_handoff_path)
        ),
        "conservative_handoffs": (
            None if conservative_handoff_path is None else str(conservative_handoff_path)
        ),
        "handoff_source": handoff_source,
    }

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    map_yaml = conservative_path.with_suffix(".yaml")
    bundle = write_headland_connectivity_bundle(
        result,
        conservative,
        output,
        source_map_yaml=(map_yaml if map_yaml.exists() else None),
    )

    counts = result["connectivity_counts"]
    print("output:", output)
    print("method:", result["method"])
    print("radius_m:", result["radius_m"])
    print("adjacent_pairs:", result["adjacent_pair_count"])
    print("pair_side_records:", result["pair_side_record_count"])
    print("baseline_connected:", counts["baseline_connected"])
    print("conservative_connected:", counts["conservative_connected"])
    print("gained_by_trusted_overlay:", counts["gained_by_trusted_overlay"])
    print("lost_by_trusted_overlay:", counts["lost_by_trusted_overlay"])
    print("width_ineligible_records:", counts["width_ineligible"])
    enabled = sum(item["enabled"] for item in bundle["planner_pairs"]["tests"])
    print("planner_pair_tests_enabled:", enabled)
    for item in result["pairs"]:
        if item["gained_by_trusted_overlay"]:
            print(
                f"GAIN {item['pair_id']} {item['side']}: "
                f"new_free={item['new_free_cell_count_in_domain']} "
                f"new_safe={item['new_safe_cell_count_in_domain']}"
            )


if __name__ == "__main__":
    main()
