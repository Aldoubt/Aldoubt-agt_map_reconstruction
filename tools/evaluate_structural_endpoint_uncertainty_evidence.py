#!/usr/bin/env python3
"""Evaluate frozen observation sufficiency inside fused structural endpoint ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse frozen ground/scan/ray grids inside uncertainty-aware fused D3.1 "
            "headland ROIs without changing the navigation map."
        )
    )
    parser.add_argument("--roi", required=True, help="structural_endpoint_uncertainty_roi.json")
    parser.add_argument("--ground-reference", required=True)
    parser.add_argument("--scan-support-count", required=True)
    parser.add_argument("--ray-support-count")
    parser.add_argument("--min-repeated-scans", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _load_masks(roi_payload, roi_path):
    root = roi_path.parent
    masks = {}
    for key, filename in (roi_payload.get("mask_files") or {}).items():
        path = root / filename
        masks[key] = np.load(path).astype(bool, copy=False)
    return masks


def _print_stats(label, stats):
    print(
        f"{label}: roi={stats['roi_cell_count']} unknown={stats['unknown_cell_count']} "
        f"no_ground={stats['unknown_no_ground_reference_cell_count']} "
        f"ground_no_observation={stats['unknown_ground_reference_no_observation_cell_count']} "
        f"single_scan={stats['unknown_single_scan_support_cell_count']} "
        f"repeated_scan={stats['unknown_repeated_scan_support_cell_count']} "
        f"ray_supported={stats['ray_supported_unknown_cell_count']}"
    )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE
    from agt_map_reconstruction.maps.structural_endpoint_uncertainty_evidence import (
        evaluate_uncertainty_roi_observation_sufficiency,
    )

    roi_path = Path(args.roi).expanduser().resolve()
    ground_path = Path(args.ground_reference).expanduser().resolve()
    scan_path = Path(args.scan_support_count).expanduser().resolve()
    ray_path = None if args.ray_support_count is None else Path(args.ray_support_count).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    roi_payload = json.loads(roi_path.read_text(encoding="utf-8"))
    map_text = (roi_payload.get("sources") or {}).get("map")
    if not map_text:
        raise ValueError("ROI payload must preserve sources.map")
    map_path = Path(map_text).expanduser().resolve()
    base = _read_pgm(map_path)
    ground = np.load(ground_path)
    scan = np.load(scan_path)
    ray = None if ray_path is None else np.load(ray_path)
    masks = _load_masks(roi_payload, roi_path)

    result = evaluate_uncertainty_roi_observation_sufficiency(
        base,
        ground,
        scan,
        masks,
        min_repeated_scans=int(args.min_repeated_scans),
        ray_support_count=ray,
    )
    result["sources"] = {
        "roi": str(roi_path),
        "map": str(map_path),
        "ground_reference": str(ground_path),
        "scan_support_count": str(scan_path),
        "ray_support_count": None if ray_path is None else str(ray_path),
    }
    (output / "structural_endpoint_uncertainty_evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    # Context view: classify UNKNOWN only inside the two conservative ROIs.
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    conservative = masks["entry_conservative_outward"] | masks["exit_conservative_outward"]
    unknown = conservative & (base == UNKNOWN_VALUE)
    finite_ground = np.isfinite(ground)
    threshold = int(args.min_repeated_scans)
    categories = [
        (unknown & ~finite_ground, (180, 0, 180)),
        (unknown & finite_ground & (scan < 1), (0, 140, 255)),
        (unknown & finite_ground & (scan >= 1) & (scan < threshold), (0, 220, 220)),
        (unknown & finite_ground & (scan >= threshold), (0, 200, 0)),
    ]
    for mask, color in categories:
        image[mask] = color
    image[masks["structurally_unresolved_cross"]] = (0, 0, 255)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (720, 140), (30, 30, 30), -1)
    legend = [
        ("magenta: UNKNOWN with no trusted ground reference", (180, 0, 180)),
        ("orange: UNKNOWN with ground but no scan observation", (0, 140, 255)),
        ("yellow: UNKNOWN with single/non-repeated scan support", (0, 220, 220)),
        ("green: UNKNOWN with repeated scan support", (0, 200, 0)),
        ("red: structurally unresolved cross strip; reported separately", (0, 0, 255)),
    ]
    for index, (text, color) in enumerate(legend):
        cv2.putText(
            display,
            text,
            (18, 30 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "structural_endpoint_uncertainty_evidence.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("min_repeated_scans:", result["min_repeated_scans"])
    _print_stats("entry_conservative", result["entry"]["conservative_outward"])
    _print_stats("entry_boundary_uncertainty", result["entry"]["boundary_uncertainty"])
    _print_stats("exit_conservative", result["exit"]["conservative_outward"])
    _print_stats("exit_boundary_uncertainty", result["exit"]["boundary_uncertainty"])
    _print_stats("structurally_unresolved_cross", result["structurally_unresolved_cross"])
    print("frozen_evidence_reused: true")
    print("unresolved_cross_strip_promoted_to_resolved: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
