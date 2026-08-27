#!/usr/bin/env python3
"""Evaluate support-count thresholds against the frozen P1-D3 endpoint metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold a ray- or unique-scan-support grid and rerun the exact frozen "
            "P1-D3 endpoint A/B metrics without editing the canonical navigation map."
        )
    )
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--support-count", required=True)
    parser.add_argument("--support-basis", choices=("ray", "scan"), required=True)
    parser.add_argument("--min-support", nargs="+", type=int, required=True)
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

    from agt_map_reconstruction.maps.support_threshold_endpoint_ab import (
        evaluate_support_thresholds,
    )

    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    count_path = Path(args.support_count).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    required = ("source_map", "source_row_band_regions", "source_handoffs")
    missing = [key for key in required if not baseline.get(key)]
    if missing:
        raise ValueError("frozen baseline envelope missing source fields: " + ", ".join(missing))

    map_path = _resolve_source(baseline["source_map"], baseline_path)
    regions_path = _resolve_source(baseline["source_row_band_regions"], baseline_path)
    handoffs_path = _resolve_source(baseline["source_handoffs"], baseline_path)
    base_map = _read_grid_pgm(map_path)
    support_count = np.load(count_path, allow_pickle=False)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    grid = regions_payload.get("grid", {})
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base_map.shape != expected_shape or support_count.shape != expected_shape:
        raise ValueError(
            f"shape mismatch: map={base_map.shape}, support={support_count.shape}, grid={expected_shape}"
        )

    rows = [item for item in regions_payload.get("regions", []) if item.get("region_class") == "row_aisle"]
    handoffs = list(handoffs_payload.get("handoffs", []))
    radius = handoffs_payload.get("radius_m")
    if radius is None:
        raise ValueError("frozen handoff bundle missing radius_m")

    result = evaluate_support_thresholds(
        base_map,
        support_count,
        rows,
        handoffs,
        resolution=float(grid["resolution"]),
        radius_m=float(radius),
        baseline_envelope=baseline,
        min_support_values=args.min_support,
        support_basis=args.support_basis,
    )
    result.update(
        {
            "source_baseline_envelope": str(baseline_path),
            "source_support_count": str(count_path),
            "source_map": str(map_path),
            "source_row_band_regions": str(regions_path),
            "source_handoffs": str(handoffs_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print("output:", output_path)
    print("support_basis:", result["support_basis"])
    for item in result["thresholds"]:
        print(f"min_support={item['min_support']} supported_cells={item['supported_cell_count']}")
        for side_name in ("entry", "exit"):
            side = item["comparison"]["sides"][side_name]
            base = side["baseline"]
            cand = side["candidate"]
            delta = side["delta"]
            print(
                f"  {side_name}: "
                f"coverage={_fmt(cand['cross_row_coverage_fraction'])} "
                f"coverage_gain={_fmt(delta['cross_row_coverage_fraction'])} "
                f"endpoint_median_m={_fmt(cand['endpoint_distance_median_m'])} "
                f"endpoint_reduction_m={_fmt(delta['endpoint_distance_reduction_m'])} "
                f"depth_m={_fmt(cand['max_outward_depth_m'])} "
                f"depth_gain_m={_fmt(delta['max_outward_depth_gain_m'])}"
            )
    print("automatic_threshold_selection: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
