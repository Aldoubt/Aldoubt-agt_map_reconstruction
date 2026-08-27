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
    entry_transition = item.get("entry_transition") or {}
    exit_transition = item.get("exit_transition") or {}
    return {
        "label": item.get("label"),
        "status": item.get("status"),
        "component_selection": item.get("component_selection"),
        "radius_m": item.get("radius_m"),
        "aisle_width_m": item.get("aisle_width_m"),
        "required_diameter_m": item.get("required_diameter_m"),
        "width_clearance_eligible": item.get("width_clearance_eligible"),
        "row_core_fraction": item.get("row_core_fraction"),
        "row_core_start_s_over_l": item.get("row_core_start_s_over_l"),
        "row_core_end_s_over_l": item.get("row_core_end_s_over_l"),
        "row_core_length_m": item.get("row_core_length_m"),
        "entry_transition_length_m": item.get("entry_transition_length_m"),
        "entry_transition_dominant_source": entry_transition.get("dominant_source"),
        "entry_transition_blocked_cell_count": entry_transition.get("blocked_cell_count"),
        "exit_transition_length_m": item.get("exit_transition_length_m"),
        "exit_transition_dominant_source": exit_transition.get("dominant_source"),
        "exit_transition_blocked_cell_count": exit_transition.get("blocked_cell_count"),
        "entry_handoff_s_over_l": entry.get("s_over_l"),
        "entry_cross_track_offset_m": entry.get("cross_track_offset_m"),
        "entry_clearance_m": entry.get("clearance_m"),
        "entry_boundary_nearest_source": entry.get("boundary_nearest_source"),
        "exit_handoff_s_over_l": exit_.get("s_over_l"),
        "exit_cross_track_offset_m": exit_.get("cross_track_offset_m"),
        "exit_clearance_m": exit_.get("clearance_m"),
        "exit_boundary_nearest_source": exit_.get("boundary_nearest_source"),
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
    width_eligible = [
        item for item in results if item.get("width_clearance_eligible") is True
    ]
    width_limited = [
        item for item in results if item.get("width_clearance_eligible") is False
    ]

    payload = {
        "schema_version": 2,
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
            "boundary_source_semantics": (
                "nearest source at the selected safe boundary cell only"
            ),
            "transition_source_semantics": (
                "dominant nearest source over clearance-blocked cells in the "
                "entry/exit transition zone"
            ),
            "status_semantics": (
                "ok means a safe component exists; it does not imply width "
                "eligibility or full-length aisle connectivity"
            ),
            "width_eligibility": "aisle_width_m >= 2 * radius_m",
            "map_editing": False,
        },
        "aisle_count": len(results),
        "ok_count": len(ok),
        "no_safe_component_count": len(no_safe),
        "fallback_component_count": len(fallback),
        "width_clearance_eligible_count": len(width_eligible),
        "width_limited_count": len(width_limited),
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
        "aisle_width_m",
        "required_diameter_m",
        "width_clearance_eligible",
        "row_core_fraction",
        "row_core_start_s_over_l",
        "row_core_end_s_over_l",
        "row_core_length_m",
        "entry_transition_length_m",
        "entry_transition_dominant_source",
        "entry_transition_blocked_cell_count",
        "exit_transition_length_m",
        "exit_transition_dominant_source",
        "exit_transition_blocked_cell_count",
        "entry_handoff_s_over_l",
        "entry_cross_track_offset_m",
        "entry_clearance_m",
        "entry_boundary_nearest_source",
        "exit_handoff_s_over_l",
        "exit_cross_track_offset_m",
        "exit_clearance_m",
        "exit_boundary_nearest_source",
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
    print("width_clearance_eligible:", len(width_eligible))
    print("width_limited:", len(width_limited))
    for item in results:
        if item.get("status") != "ok":
            print(
                f"{item['label']}: status={item.get('status')} "
                f"width_eligible={item.get('width_clearance_eligible')}"
            )
            continue
        entry = item["entry_handoff"]
        exit_ = item["exit_handoff"]
        entry_transition = item["entry_transition"]
        exit_transition = item["exit_transition"]
        print(
            f"{item['label']}: width_eligible={item['width_clearance_eligible']} "
            f"core_fraction={item['row_core_fraction']:.3f} "
            f"core={item['row_core_start_s_over_l']:.3f}.."
            f"{item['row_core_end_s_over_l']:.3f} "
            f"entry_transition_m={item['entry_transition_length_m']:.2f} "
            f"exit_transition_m={item['exit_transition_length_m']:.2f} "
            f"entry_offset_m={entry['cross_track_offset_m']:.2f} "
            f"exit_offset_m={exit_['cross_track_offset_m']:.2f} "
            f"entry_boundary_source={entry['boundary_nearest_source']} "
            f"exit_boundary_source={exit_['boundary_nearest_source']} "
            f"entry_transition_source={entry_transition['dominant_source']} "
            f"exit_transition_source={exit_transition['dominant_source']}"
        )


if __name__ == "__main__":
    main()
