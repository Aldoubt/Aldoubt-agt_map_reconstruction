#!/usr/bin/env python3
"""Analyze row-handoff connectivity to wide open-area candidates."""

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
            "Classify each row-core handoff by global connected-component reachability "
            "to wide open-area candidates. This is not a side-local headland test."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument("--row-band-regions", required=True, help="row_band_regions.json")
    parser.add_argument("--handoffs", required=True, help="aisle_handoffs.json")
    parser.add_argument("--output", required=True)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _csv_row(item):
    return {
        "label": item.get("label"),
        "side": item.get("side"),
        "radius_m": item.get("radius_m"),
        "width_clearance_eligible": item.get("width_clearance_eligible"),
        "connectivity_scope": "global_component",
        "connectivity_class": item.get("connectivity_class"),
        "strict_connected_candidates": ";".join(item.get("strict_connected_candidates", [])),
        "unknown_bridge_candidates": ";".join(item.get("unknown_bridge_candidates", [])),
        "nearest_open_area_label": item.get("nearest_open_area_label"),
        "nearest_open_area_distance_m": item.get("nearest_open_area_distance_m"),
    }


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.open_area_topology import (
        analyze_handoff_open_area_topology,
    )

    map_path = Path(args.map).expanduser().resolve()
    regions_path = Path(args.row_band_regions).expanduser().resolve()
    handoffs_path = Path(args.handoffs).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_map = _read_grid_pgm(map_path)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))

    grid = regions_payload.get("grid", {})
    resolution = float(grid["resolution"])
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base_map.shape != expected_shape:
        raise ValueError(
            f"map shape {base_map.shape} does not match region grid {expected_shape}"
        )

    radius = handoffs_payload.get("radius_m")
    if radius is None:
        raise ValueError("handoff bundle must contain one global radius_m")

    open_regions = [
        item
        for item in regions_payload.get("regions", [])
        if item.get("region_class") == "wide_open_area_candidate"
    ]
    handoffs = list(handoffs_payload.get("handoffs", []))
    result = analyze_handoff_open_area_topology(
        base_map,
        handoffs,
        open_regions,
        resolution=resolution,
        radius_m=float(radius),
    )
    result.update({
        "connectivity_scope": "global_component",
        "scope_note": (
            "A connected handoff and candidate share one global traversable component. "
            "This does not prove that the candidate lies beyond that specific entry/exit side."
        ),
        "source_map": str(map_path),
        "source_row_band_regions": str(regions_path),
        "source_handoffs": str(handoffs_path),
    })

    json_path = output / "open_area_topology.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = [_csv_row(item) for item in result["handoffs"]]
    fieldnames = list(rows[0]) if rows else [
        "label",
        "side",
        "radius_m",
        "width_clearance_eligible",
        "connectivity_scope",
        "connectivity_class",
        "strict_connected_candidates",
        "unknown_bridge_candidates",
        "nearest_open_area_label",
        "nearest_open_area_distance_m",
    ]
    with (output / "open_area_topology.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("output:", output)
    print(f"radius: {float(radius):.2f}")
    print("connectivity_scope: global_component")
    print("handoffs:", result["handoff_count"])
    print("open_area_candidates:", result["open_area_candidate_count"])
    print("connectivity_counts:", result["connectivity_counts"])
    for region in result["open_area_candidates"]:
        print(
            f"{region['label']}: strict={region['strict_connection_count']} "
            f"unknown_bridge={region['unknown_bridge_connection_count']} "
            f"strict_entry={region['strict_entry_aisles']} "
            f"strict_exit={region['strict_exit_aisles']} "
            f"unknown_entry={region['unknown_bridge_entry_aisles']} "
            f"unknown_exit={region['unknown_bridge_exit_aisles']}"
        )


if __name__ == "__main__":
    main()
