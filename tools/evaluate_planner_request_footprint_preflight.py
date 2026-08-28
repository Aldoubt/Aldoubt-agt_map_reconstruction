#!/usr/bin/env python3
"""Evaluate polygon-footprint validity of frozen planner request poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.planner_request_footprint_preflight import (
    audit_planner_request_footprints,
    write_planner_request_footprint_preflight_bundle,
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
            "Read-only P1-G1.1 polygon-footprint preflight for frozen Nav2 "
            "planner request start/goal poses."
        )
    )
    parser.add_argument("--map-pgm", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--planner-pairs", required=True)
    parser.add_argument("--gap-diagnostics", required=True)
    parser.add_argument("--footprint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pgm_path, yaml_path, base_map, metadata = _load_map(
        args.map_pgm, args.map_yaml
    )
    planner_pairs_path, planner_pairs = _load_payload(args.planner_pairs)
    diagnostics_path, diagnostics = _load_payload(args.gap_diagnostics)
    footprint_path, footprint_payload = _load_payload(args.footprint)

    if "polygon_xy_m" not in footprint_payload:
        raise ValueError("footprint artifact requires polygon_xy_m")

    result = audit_planner_request_footprints(
        base_map,
        planner_pairs,
        diagnostics,
        footprint_payload["polygon_xy_m"],
        metadata,
        footprint_name=footprint_payload.get("name", "robot"),
    )
    result["sources"] = {
        "map_pgm": str(pgm_path),
        "map_yaml": str(yaml_path),
        "planner_pairs": str(planner_pairs_path),
        "gap_diagnostics": str(diagnostics_path),
        "footprint": str(footprint_path),
    }

    output = Path(args.output).expanduser().resolve()
    write_planner_request_footprint_preflight_bundle(result, output)

    summary = result["summary"]
    print("output:", output)
    print("method:", result["method"])
    print("request_count:", summary["request_count"])
    print("pose_count:", summary["pose_count"])
    print("request_valid_count:", summary["request_valid_count"])
    print("start_invalid_request_count:", summary["start_invalid_request_count"])
    print("goal_invalid_request_count:", summary["goal_invalid_request_count"])
    print("pose_class_counts:", summary["pose_class_counts"])

    for item in result["requests"]:
        if item["request_valid"]:
            continue
        print(
            f"{item['request_id']}: "
            f"start={item['start_preflight']['pose_class']} "
            f"goal={item['goal_preflight']['pose_class']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
