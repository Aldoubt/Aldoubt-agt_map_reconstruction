#!/usr/bin/env python3
"""Evaluate a ray-support mask against the frozen P1-D3 endpoint envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use the exact frozen P1-D3 source map/row/handoff geometry and treat only "
            "ray-supported UNKNOWN cells as observed free in an evaluation-only overlay."
        )
    )
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--ray-support-mask", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _resolve_source(value, baseline_path):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = baseline_path.parent / path
    return path.resolve()


def _fmt(value):
    return "None" if value is None else f"{float(value):.6f}"


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.ray_endpoint_evidence_ab import (
        evaluate_ray_supported_endpoint_envelope,
    )

    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    support_path = Path(args.ray_support_mask).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    required_sources = ("source_map", "source_row_band_regions", "source_handoffs")
    missing = [key for key in required_sources if not baseline.get(key)]
    if missing:
        raise ValueError("frozen baseline envelope missing source fields: " + ", ".join(missing))

    map_path = _resolve_source(baseline["source_map"], baseline_path)
    regions_path = _resolve_source(baseline["source_row_band_regions"], baseline_path)
    handoffs_path = _resolve_source(baseline["source_handoffs"], baseline_path)

    base_map = _read_grid_pgm(map_path)
    support = np.load(support_path, allow_pickle=False).astype(bool)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    grid = regions_payload.get("grid", {})
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base_map.shape != expected_shape or support.shape != expected_shape:
        raise ValueError(
            f"shape mismatch: map={base_map.shape}, support={support.shape}, grid={expected_shape}"
        )

    rows = [
        item
        for item in regions_payload.get("regions", [])
        if item.get("region_class") == "row_aisle"
    ]
    handoffs = list(handoffs_payload.get("handoffs", []))
    radius = handoffs_payload.get("radius_m")
    if radius is None:
        raise ValueError("frozen handoff bundle missing radius_m")

    result = evaluate_ray_supported_endpoint_envelope(
        base_map,
        support,
        rows,
        handoffs,
        resolution=float(grid["resolution"]),
        radius_m=float(radius),
        baseline_envelope=baseline,
    )
    result.update({
        "schema_version": 1,
        "source_baseline_envelope": str(baseline_path),
        "source_ray_support_mask": str(support_path),
        "source_map": str(map_path),
        "source_row_band_regions": str(regions_path),
        "source_handoffs": str(handoffs_path),
    })
    (output / "ray_endpoint_evidence_ab.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    overlay = result["overlay_summary"]
    print("output:", output)
    print("ray_supported_cells:", overlay["ray_supported_cell_count"])
    print("ray_supported_unknown_cells:", overlay["ray_supported_unknown_cell_count"])
    print(
        "ray_supported_occupied_cells_ignored:",
        overlay["ray_supported_occupied_cell_count_ignored"],
    )
    for side_name in ("entry", "exit"):
        side = result["comparison"]["sides"][side_name]
        base = side["baseline"]
        candidate = side["candidate"]
        delta = side["delta"]
        print(
            f"{side_name}: "
            f"baseline_coverage={_fmt(base['cross_row_coverage_fraction'])} "
            f"candidate_coverage={_fmt(candidate['cross_row_coverage_fraction'])} "
            f"coverage_gain={_fmt(delta['cross_row_coverage_fraction'])} "
            f"baseline_endpoint_median_m={_fmt(base['endpoint_distance_median_m'])} "
            f"candidate_endpoint_median_m={_fmt(candidate['endpoint_distance_median_m'])} "
            f"endpoint_reduction_m={_fmt(delta['endpoint_distance_reduction_m'])} "
            f"baseline_depth_m={_fmt(base['max_outward_depth_m'])} "
            f"candidate_depth_m={_fmt(candidate['max_outward_depth_m'])} "
            f"depth_gain_m={_fmt(delta['max_outward_depth_gain_m'])}"
        )
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
