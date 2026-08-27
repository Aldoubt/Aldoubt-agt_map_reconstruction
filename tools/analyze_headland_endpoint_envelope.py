#!/usr/bin/env python3
"""Analyze endpoint-side free-space envelopes for headland evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze strict-free and unknown-relaxed evidence outside the common "
            "entry/exit endpoint lines of clearance-width eligible row aisles."
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


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def _side_summary(side):
    strict = side["strict"].get("best_component")
    relaxed = side["relaxed_unknown_allowed"].get("best_component")
    gap = side.get("evidence_gap") or {}
    return {
        "strict_coverage": None if strict is None else strict["cross_row_coverage_fraction"],
        "strict_endpoint_median_m": None if strict is None else strict["endpoint_distance_median_m"],
        "strict_depth_m": None if strict is None else strict["max_outward_depth_m"],
        "relaxed_coverage": None if relaxed is None else relaxed["cross_row_coverage_fraction"],
        "relaxed_endpoint_median_m": None if relaxed is None else relaxed["endpoint_distance_median_m"],
        "relaxed_depth_m": None if relaxed is None else relaxed["max_outward_depth_m"],
        "relaxed_unknown_fraction": None if relaxed is None else relaxed["unknown_cell_fraction"],
        "coverage_gain": gap.get("coverage_gain"),
        "endpoint_distance_reduction_m": gap.get("endpoint_distance_reduction_m"),
        "outward_depth_gain_m": gap.get("outward_depth_gain_m"),
        "relaxed_observed_fraction": gap.get("relaxed_observed_fraction"),
    }


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_endpoint_envelope import (
        analyze_endpoint_side_envelopes,
    )
    from agt_map_reconstruction.maps.headland_evidence_gap import (
        attach_endpoint_evidence_gaps,
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

    rows = [
        item
        for item in regions_payload.get("regions", [])
        if item.get("region_class") == "row_aisle"
    ]
    handoffs = list(handoffs_payload.get("handoffs", []))
    result = analyze_endpoint_side_envelopes(
        base_map,
        rows,
        handoffs,
        resolution=resolution,
        radius_m=float(radius),
    )
    attach_endpoint_evidence_gaps(result)
    result.update({
        "source_map": str(map_path),
        "source_row_band_regions": str(regions_path),
        "source_handoffs": str(handoffs_path),
    })

    json_path = output / "headland_endpoint_envelope.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print(f"radius: {float(radius):.2f}")
    print("eligible_rows:", result["eligible_row_count"])
    print("eligible_labels:", result["eligible_row_labels"])
    print("row_axis_direction:", result["row_axis_direction"])
    print("row_cross_span:", result["row_cross_span"])
    for side_name in ("entry", "exit"):
        summary = _side_summary(result["sides"][side_name])
        print(
            f"{side_name}: "
            f"strict_coverage={_fmt(summary['strict_coverage'])} "
            f"strict_endpoint_median_m={_fmt(summary['strict_endpoint_median_m'])} "
            f"strict_depth_m={_fmt(summary['strict_depth_m'])} "
            f"relaxed_coverage={_fmt(summary['relaxed_coverage'])} "
            f"relaxed_endpoint_median_m={_fmt(summary['relaxed_endpoint_median_m'])} "
            f"relaxed_depth_m={_fmt(summary['relaxed_depth_m'])} "
            f"relaxed_unknown_fraction={_fmt(summary['relaxed_unknown_fraction'])} "
            f"coverage_gain={_fmt(summary['coverage_gain'])} "
            f"endpoint_reduction_m={_fmt(summary['endpoint_distance_reduction_m'])} "
            f"depth_gain_m={_fmt(summary['outward_depth_gain_m'])} "
            f"relaxed_observed_fraction={_fmt(summary['relaxed_observed_fraction'])}"
        )


if __name__ == "__main__":
    main()
