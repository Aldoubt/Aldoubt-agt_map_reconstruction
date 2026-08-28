#!/usr/bin/env python3
"""Audit Nav2 planner paths against frozen P1 pair-side topology scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Classify Nav2 planner paths as local pair-domain matches, global "
            "detours, 8-connected local matches, or remaining clearance mismatches."
        )
    )
    parser.add_argument("--map", required=True, help="Conservative navigation PGM")
    parser.add_argument("--aisles", required=True, help="aisle_rectangles.json")
    parser.add_argument("--connectivity", required=True, help="headland_connectivity.json")
    parser.add_argument("--depth-profile", required=True, help="headland_depth_profile.json")
    parser.add_argument("--planner-results", required=True, help="planner_smoke_results.json")
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--output", required=True)
    return parser


def _read_map(path):
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read map: {path}")
    return path, np.flipud(image).astype(np.uint8, copy=False)


def _read_json(path, label):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, payload


def _load_aisles(path):
    path, payload = _read_json(path, "aisles")
    aisles = payload.get("rectangles")
    grid = dict(payload.get("grid") or {})
    if not isinstance(aisles, list):
        raise ValueError("aisle JSON must contain rectangles")
    for key in ("resolution", "origin", "width", "height"):
        if key not in grid:
            raise ValueError(f"aisle grid.{key} is required")
    return path, aisles, grid


def _load_depth_profile(path):
    path, payload = _read_json(path, "depth profile")
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("depth profile must contain mask_files")
    masks = {
        key: np.load(path.parent / filename, allow_pickle=False).astype(bool, copy=False)
        for key, filename in files.items()
    }
    return path, payload, masks


def _metadata(grid):
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    from agt_map_reconstruction.maps.planner_path_scope_audit import (
        analyze_planner_path_scope_audit,
        write_planner_path_scope_audit_bundle,
    )

    map_path, base_map = _read_map(args.map)
    aisle_path, aisles, grid = _load_aisles(args.aisles)
    connectivity_path, connectivity = _read_json(args.connectivity, "connectivity")
    profile_path, profile, masks = _load_depth_profile(args.depth_profile)
    planner_path, planner_results = _read_json(args.planner_results, "planner results")
    metadata = _metadata(grid)

    expected_shape = (int(metadata.height), int(metadata.width))
    if base_map.shape != expected_shape:
        raise ValueError(f"map shape {base_map.shape} does not match aisle grid {expected_shape}")

    audit = analyze_planner_path_scope_audit(
        base_map,
        aisles,
        connectivity,
        profile,
        masks,
        planner_results,
        metadata=metadata,
        resolution=float(metadata.resolution),
        radius_m=float(args.radius),
    )
    audit["sources"] = {
        "conservative_map": str(map_path),
        "aisles": str(aisle_path),
        "connectivity": str(connectivity_path),
        "depth_profile": str(profile_path),
        "planner_results": str(planner_path),
    }

    output = Path(args.output).expanduser().resolve()
    write_planner_path_scope_audit_bundle(audit, output)

    summary = audit["summary"]
    print("output:", output)
    print("method:", audit["method"])
    print("radius_m:", audit["radius_m"])
    print("records:", summary["record_count"])
    print("planner_success:", summary["planner_success"])
    print("planner_failure:", summary["planner_failure"])
    print("infrastructure_error:", summary["infrastructure_error"])
    print("strict4_contract_mismatch:", summary["strict4_contract_mismatch"])
    print("classification_counts:")
    for key, count in sorted(summary["classification_counts"].items()):
        print(f"  {key}: {count}")
    print("scope_counts:")
    for key, count in sorted(summary["scope_counts"].items()):
        print(f"  {key}: {count}")

    interesting = {
        "negative_global_detour",
        "negative_local_8connect_match",
        "negative_local_clearance_mismatch",
        "frozen_topology_recompute_mismatch",
    }
    for item in audit["records"]:
        if item["classification"] not in interesting:
            continue
        clearance = item["min_source_map_clearance_m"]
        clearance_text = "n/a" if clearance is None else f"{clearance:.3f}m"
        ratio = item["detour_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
        print(
            f"{item['request_id']}: class={item['classification']} "
            f"scope={item['scope_class']} strict4={item['strict_connected_4']} "
            f"strict8={item['strict_connected_8']} ratio={ratio_text} "
            f"outside_pair={item['path_outside_pair_domain_fraction']:.3f} "
            f"outside_finite={item['path_outside_finite_headland_fraction']:.3f} "
            f"min_clearance={clearance_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
