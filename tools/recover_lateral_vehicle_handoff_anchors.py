#!/usr/bin/env python3
"""Recover lateral-aware vehicle-feasible handoff anchors from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.vehicle_handoff_anchor_lateral_recovery import (
    recover_lateral_vehicle_handoff_anchors,
    write_lateral_vehicle_handoff_anchor_recovery_bundle,
)


def _load_payload(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain an object: {path}")
    return path, payload


def _load_map(pgm_path, yaml_path):
    pgm = Path(pgm_path).expanduser().resolve()
    map_yaml = Path(yaml_path).expanduser().resolve()
    config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("map YAML must contain an object")

    image = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(pgm)
    grid = np.flipud(image).astype(np.uint8, copy=False)

    resolution = float(config["resolution"])
    origin = list(config["origin"])
    if resolution <= 0.0 or len(origin) < 3:
        raise ValueError("map YAML resolution/origin is invalid")
    metadata = GridMetadata(
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=int(grid.shape[1]),
        height=int(grid.shape[0]),
        frame_id="map",
    )
    return pgm, map_yaml, grid, metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only P1-G1.2b recovery of vehicle-feasible handoff anchors. "
            "Search uses the frozen map resolution, moves inward only along the "
            "aisle, permits lateral motion only inside the aisle footprint-feasible "
            "band, and keeps yaw fixed."
        )
    )
    parser.add_argument("--map-pgm", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--aisles", required=True)
    parser.add_argument("--planner-pairs", required=True)
    parser.add_argument("--footprint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pgm_path, yaml_path, base_map, metadata = _load_map(args.map_pgm, args.map_yaml)
    aisles_path, aisles_payload = _load_payload(args.aisles)
    planner_pairs_path, planner_pairs = _load_payload(args.planner_pairs)
    footprint_path, footprint_payload = _load_payload(args.footprint)

    rectangles = aisles_payload.get("rectangles")
    if not isinstance(rectangles, list):
        raise ValueError("aisle artifact requires rectangles list")
    if "polygon_xy_m" not in footprint_payload:
        raise ValueError("footprint artifact requires polygon_xy_m")

    result = recover_lateral_vehicle_handoff_anchors(
        base_map,
        rectangles,
        planner_pairs,
        footprint_payload["polygon_xy_m"],
        metadata,
        footprint_name=footprint_payload.get("name", "robot"),
    )
    result["sources"] = {
        "map_pgm": str(pgm_path),
        "map_yaml": str(yaml_path),
        "aisles": str(aisles_path),
        "planner_pairs": str(planner_pairs_path),
        "footprint": str(footprint_path),
    }

    output = Path(args.output).expanduser().resolve()
    write_lateral_vehicle_handoff_anchor_recovery_bundle(result, output)

    summary = result["summary"]
    print("output:", output)
    print("method:", result["method"])
    print("anchor_count:", summary["anchor_count"])
    print("available_count:", summary["available_count"])
    print("unavailable_count:", summary["unavailable_count"])
    print("already_valid_count:", summary["already_valid_count"])
    print("recovered_longitudinal_count:", summary["recovered_longitudinal_count"])
    print("recovered_lateral_count:", summary["recovered_lateral_count"])
    print(
        "improved_over_longitudinal_only_count:",
        summary["improved_over_longitudinal_only_count"],
    )
    print(
        "recovered_from_longitudinal_unavailable_count:",
        summary["recovered_from_longitudinal_unavailable_count"],
    )

    for item in result["anchors"]:
        print(
            f"{item['anchor_id']}: status={item['recovery_status']} "
            f"long_only={item['longitudinal_only_status']} "
            f"long_inset={item['longitudinal_only_inset_m']} "
            f"inset={item['longitudinal_inset_m']} "
            f"lateral={item['lateral_shift_m']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
