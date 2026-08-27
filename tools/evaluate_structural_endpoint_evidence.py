#!/usr/bin/env python3
"""Re-evaluate frozen E0/E1/E2 evidence in the P1-D3.1 structural ROI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen navigation map, ground-reference grid and scan/ray "
            "support arrays against D3.1 structural endpoint geometry without "
            "replaying rosbag or modifying the canonical map."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument("--structural-boundary", required=True)
    parser.add_argument("--ground-reference", required=True)
    parser.add_argument("--scan-support-count", required=True)
    parser.add_argument("--ray-support-count")
    parser.add_argument("--min-repeated-scans", type=int, default=2)
    parser.add_argument("--radius-m", type=float, default=0.20)
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
    return f"{float(value):.6f}"


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_roi import (
        build_structural_endpoint_roi,
        evaluate_structural_endpoint_evidence,
    )

    map_path = Path(args.map).expanduser().resolve()
    boundary_path = Path(args.structural_boundary).expanduser().resolve()
    ground_path = Path(args.ground_reference).expanduser().resolve()
    scan_path = Path(args.scan_support_count).expanduser().resolve()
    ray_path = (
        None
        if args.ray_support_count is None
        else Path(args.ray_support_count).expanduser().resolve()
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_grid_pgm(map_path)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    ground = np.load(ground_path, allow_pickle=False)
    scan = np.load(scan_path, allow_pickle=False)
    ray = None if ray_path is None else np.load(ray_path, allow_pickle=False)

    result = evaluate_structural_endpoint_evidence(
        base,
        boundary,
        ground_reference=ground,
        scan_support_count=scan,
        ray_support_count=ray,
        min_repeated_scans=int(args.min_repeated_scans),
        radius_m=float(args.radius_m),
    )
    result["sources"] = {
        "map": str(map_path),
        "structural_boundary": str(boundary_path),
        "ground_reference": str(ground_path),
        "scan_support_count": str(scan_path),
        "ray_support_count": None if ray_path is None else str(ray_path),
    }

    for side in ("entry", "exit"):
        roi = build_structural_endpoint_roi(base.shape, boundary, side)
        np.save(output / f"{side}_structural_endpoint_roi.npy", roi)

    json_path = output / "structural_endpoint_evidence.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("radius_m:", float(args.radius_m))
    print("min_repeated_scans:", int(args.min_repeated_scans))
    for side in ("entry", "exit"):
        item = result[side]
        evidence = item["evidence"]
        delta = item["delta"]
        baseline = item["baseline_strict"].get("best_component")
        candidate = item["candidate_repeated_scan_overlay"].get("best_component")
        print(
            f"{side}: roi={evidence['roi_cell_count']} "
            f"unknown={evidence['unknown_cell_count']} "
            f"ground_finite_unknown={evidence['ground_finite_unknown_cell_count']} "
            f"scan_supported_unknown={evidence['repeated_scan_supported_unknown_cell_count']}"
        )
        print(
            f"  baseline_coverage={_fmt(None if baseline is None else baseline['cross_row_coverage_fraction'])} "
            f"candidate_coverage={_fmt(None if candidate is None else candidate['cross_row_coverage_fraction'])} "
            f"coverage_gain={_fmt(delta['coverage_gain'])}"
        )
        print(
            f"  baseline_endpoint_median_m={_fmt(None if baseline is None else baseline['endpoint_distance_median_m'])} "
            f"candidate_endpoint_median_m={_fmt(None if candidate is None else candidate['endpoint_distance_median_m'])} "
            f"endpoint_reduction_m={_fmt(delta['endpoint_distance_reduction_m'])}"
        )
        print(
            f"  baseline_depth_m={_fmt(None if baseline is None else baseline['max_outward_depth_m'])} "
            f"candidate_depth_m={_fmt(None if candidate is None else candidate['max_outward_depth_m'])} "
            f"depth_gain_m={_fmt(delta['outward_depth_gain_m'])}"
        )
    print("frozen_evidence_reused: true")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
