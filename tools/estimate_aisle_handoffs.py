#!/usr/bin/env python3
"""Estimate clearance-conditioned row-core handoff poses for recovered aisles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate entry/exit handoff boundaries from the connected safe row core "
            "without editing the static navigation map."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument("--aisles", required=True, help="aisle_rectangles.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radius",
        type=float,
        default=0.20,
        help="Clearance radius in metres used to define the safe row core.",
    )
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _summary_row(item):
    entry = item.get("entry_handoff") or {}
    exit_ = item.get("exit_handoff") or {}
    return {
        "label": item.get("label"),
        "status": item.get("status"),
        "component_selection": item.get("component_selection"),
        "radius_m": item.get("radius_m"),
        "row_core_start_s_over_l": item.get("row_core_start_s_over_l"),
        "row_core_end_s_over_l": item.get("row_core_end_s_over_l"),
        "row_core_length_m": item.get("row_core_length_m"),
        "entry_transition_length_m": item.get("entry_transition_length_m"),
        "exit_transition_length_m": item.get("exit_transition_length_m"),
        "entry_handoff_s_over_l": entry.get("s_over_l"),
        "entry_cross_track_offset_m": entry.get("cross_track_offset_m"),
        "entry_clearance_m": entry.get("clearance_m"),
        "entry_boundary_source": entry.get("boundary_source"),
        "exit_handoff_s_over_l": exit_.get("s_over_l"),
        "exit_cross_track_offset_m": exit_.get("cross_track_offset_m"),
        "exit_clearance_m": exit_.get("clearance_m"),
        "exit_boundary_source": exit_.get("boundary_source"),
    }


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


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.aisle_handoff_boundary import (
        estimate_aisle_handoff_boundary,
    )

    map_path = Path(args.map).expanduser().resolve()
    aisle_path = Path(args.aisles).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_map = _read_grid_pgm(map_path)
    aisle_payload = json.loads(aisle_path.read_text(encoding="utf-8"))
    grid = aisle_payload.get("grid", {})
    metadata = _metadata_from_grid(grid)
    resolution = float(metadata.resolution)
    expected_shape = (int(metadata.height), int(metadata.width))
    if base_map.shape != expected_shape:
        raise ValueError(
            f"map shape {base_map.shape} does not match aisle grid {expected_shape}"
        )

    aisles = list(aisle_payload.get("rectangles", []))
    results = [
        estimate_aisle_handoff_boundary(
            base_map,
            aisle,
            resolution=resolution,
            radius_m=float(args.radius),
            metadata=metadata,
        )
        for aisle in aisles
    ]

    ok = [item for item in results if item.get("status") == "ok"]
    no_safe = [
        item for item in results if item.get("status") == "no_safe_component"
    ]
    fallback = [
        item for item in ok
        if item.get("component_selection") == "largest_longitudinal_span"
    ]

    payload = {
        "schema_version": 1,
        "source_map": str(map_path),
        "source_aisles": str(aisle_path),
        "grid": metadata.to_dict(),
        "resolution_m": resolution,
        "radius_m": float(args.radius),
        "policy": {
            "safe_definition": "free && distance_to_nonfree >= radius",
            "component_selection": (
                "midpoint component; fallback to largest longitudinal span"
            ),
            "handoff_pose": (
                "actual safe component boundary cell; maximize clearance then "
                "minimize absolute cross-track offset"
            ),
            "map_editing": False,
        },
        "aisle_count": len(results),
        "ok_count": len(ok),
        "no_safe_component_count": len(no_safe),
        "fallback_component_count": len(fallback),
        "handoffs": results,
    }
    (output / "aisle_handoffs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = [_summary_row(item) for item in results]
    fieldnames = list(rows[0]) if rows else [
        "label",
        "status",
        "component_selection",
        "radius_m",
        "row_core_start_s_over_l",
        "row_core_end_s_over_l",
        "row_core_length_m",
        "entry_transition_length_m",
        "exit_transition_length_m",
        "entry_handoff_s_over_l",
        "entry_cross_track_offset_m",
        "entry_clearance_m",
        "entry_boundary_source",
        "exit_handoff_s_over_l",
        "exit_cross_track_offset_m",
        "exit_clearance_m",
        "exit_boundary_source",
    ]
    with (output / "aisle_handoffs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("output:", output)
    print(f"radius: {float(args.radius):.2f}")
    print("aisles:", len(results))
    print("ok:", len(ok))
    print("no_safe_component:", len(no_safe))
    print("largest_span_fallback:", len(fallback))
    for item in results:
        if item.get("status") != "ok":
            print(f"{item['label']}: status={item.get('status')}")
            continue
        entry = item["entry_handoff"]
        exit_ = item["exit_handoff"]
        print(
            f"{item['label']}: core={item['row_core_start_s_over_l']:.3f}.."
            f"{item['row_core_end_s_over_l']:.3f} "
            f"entry_transition_m={item['entry_transition_length_m']:.2f} "
            f"exit_transition_m={item['exit_transition_length_m']:.2f} "
            f"entry_offset_m={entry['cross_track_offset_m']:.2f} "
            f"exit_offset_m={exit_['cross_track_offset_m']:.2f} "
            f"entry_source={entry['boundary_source']} "
            f"exit_source={exit_['boundary_source']}"
        )


if __name__ == "__main__":
    main()
