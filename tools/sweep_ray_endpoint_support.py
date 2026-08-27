#!/usr/bin/env python3
"""Sweep ray support-count thresholds inside the frozen P1-D3 endpoint ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Localize ray-supported UNKNOWN cells in the exact frozen P1-D3 entry/exit "
            "ROIs and report how support-count thresholds affect strict-safe geometry."
        )
    )
    parser.add_argument("--baseline-envelope", required=True)
    parser.add_argument("--ray-support-count", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support-rays", nargs="+", type=int, required=True)
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


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.ray_endpoint_support_diagnostics import (
        sweep_endpoint_support_thresholds,
    )

    baseline_path = Path(args.baseline_envelope).expanduser().resolve()
    count_path = Path(args.ray_support_count).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not baseline.get("source_map") or not baseline.get("source_row_band_regions"):
        raise ValueError("baseline envelope missing frozen source_map/source_row_band_regions")
    map_path = _resolve_source(baseline["source_map"], baseline_path)
    regions_path = _resolve_source(baseline["source_row_band_regions"], baseline_path)

    base_map = _read_grid_pgm(map_path)
    support_count = np.load(count_path, allow_pickle=False)
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    grid = regions.get("grid", {})
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base_map.shape != expected_shape or support_count.shape != expected_shape:
        raise ValueError(
            f"shape mismatch: map={base_map.shape}, support_count={support_count.shape}, "
            f"grid={expected_shape}"
        )

    result = sweep_endpoint_support_thresholds(
        base_map,
        support_count,
        baseline,
        min_support_values=args.min_support_rays,
        resolution=float(grid["resolution"]),
    )
    result.update({
        "source_baseline_envelope": str(baseline_path),
        "source_ray_support_count": str(count_path),
        "source_map": str(map_path),
        "source_row_band_regions": str(regions_path),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print("output:", output_path)
    print("radius_m:", result["radius_m"])
    for item in result["thresholds"]:
        print(
            f"min_support={item['min_support_rays']} "
            f"supported_unknown={item['supported_unknown_cell_count']} "
            f"new_strict_safe={item['new_strict_safe_cell_count']}"
        )
        for side_name in ("entry", "exit"):
            side = item["sides"][side_name]
            print(
                f"  {side_name}: "
                f"supported_unknown={side['supported_unknown_cell_count']} "
                f"roi_unknown_fraction={side['supported_unknown_fraction_of_roi_unknown']:.6f} "
                f"components={side['component_count']} "
                f"largest_component={side['largest_component_cell_count']} "
                f"raw_cross_span={side['raw_supported_cross_row_span_fraction']:.6f} "
                f"raw_depth_m={side['raw_supported_max_outward_depth_m']:.6f} "
                f"new_strict_safe={side['new_strict_safe_cell_count']}"
            )
    print("automatic_threshold_selection: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
