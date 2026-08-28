#!/usr/bin/env python3
"""Diagnose remaining scoped headland gaps after conservative free promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Classify why adjacent-aisle headland handoffs remain disconnected "
            "under a fixed clearance radius without editing the map."
        )
    )
    parser.add_argument("--baseline-map", required=True)
    parser.add_argument("--conservative-map", required=True)
    parser.add_argument("--aisles", required=True)
    parser.add_argument("--depth-profile", required=True)
    parser.add_argument("--connectivity", required=True)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--output", required=True)
    return parser


def _read_pgm(path):
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read map: {path}")
    return path, np.flipud(image).astype(np.uint8, copy=False)


def _load_aisles(path):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rectangles = payload.get("rectangles") if isinstance(payload, dict) else None
    if not isinstance(rectangles, list):
        raise ValueError("aisle JSON must contain rectangles")
    grid = dict(payload.get("grid") or {})
    if "resolution" not in grid:
        raise ValueError("aisle JSON grid.resolution is required")
    return path, rectangles, grid


def _load_depth_profile(path):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = dict(payload.get("mask_files") or {})
    if not files:
        raise ValueError("depth profile must contain mask_files")
    masks = {
        key: np.load(path.parent / filename, allow_pickle=False).astype(bool, copy=False)
        for key, filename in files.items()
    }
    return path, payload, masks


def _load_json(path):
    path = Path(path).expanduser().resolve()
    return path, json.loads(path.read_text(encoding="utf-8"))


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_gap_diagnostics import (
        analyze_headland_gap_diagnostics,
        write_headland_gap_diagnostics_bundle,
    )

    baseline_path, baseline = _read_pgm(args.baseline_map)
    conservative_path, conservative = _read_pgm(args.conservative_map)
    aisle_path, aisles, grid = _load_aisles(args.aisles)
    profile_path, profile, masks = _load_depth_profile(args.depth_profile)
    connectivity_path, connectivity = _load_json(args.connectivity)

    if baseline.shape != conservative.shape:
        raise ValueError("baseline and conservative maps must have the same shape")
    expected_shape = (
        int(grid.get("height", baseline.shape[0])),
        int(grid.get("width", baseline.shape[1])),
    )
    if baseline.shape != expected_shape:
        raise ValueError(
            f"map shape {baseline.shape} does not match aisle grid {expected_shape}"
        )
    radius = float(args.radius)
    connectivity_radius = connectivity.get("radius_m")
    if connectivity_radius is not None and not np.isclose(
        float(connectivity_radius), radius
    ):
        raise ValueError(
            f"connectivity radius {connectivity_radius} does not match requested radius {radius}"
        )

    result = analyze_headland_gap_diagnostics(
        baseline,
        conservative,
        aisles,
        connectivity,
        profile,
        masks,
        resolution=float(grid["resolution"]),
        radius_m=radius,
    )
    result["sources"] = {
        "baseline_map": str(baseline_path),
        "conservative_map": str(conservative_path),
        "aisles": str(aisle_path),
        "depth_profile": str(profile_path),
        "connectivity": str(connectivity_path),
    }

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_headland_gap_diagnostics_bundle(result, conservative, output)

    print("output:", output)
    print("method:", result["method"])
    print("radius_m:", result["radius_m"])
    print("records:", result["record_count"])
    print("evaluated_records:", result["evaluated_record_count"])
    for key in sorted(result["failure_counts"]):
        print(f"{key}: {result['failure_counts'][key]}")
    bridge_counts = {}
    for item in result["records"]:
        key = item["bridge_class"]
        bridge_counts[key] = bridge_counts.get(key, 0) + 1
    for key in sorted(bridge_counts):
        print(f"{key}: {bridge_counts[key]}")

    print("\nfailed evaluated records:")
    for item in result["records"]:
        if item["evaluation_status"] != "evaluated" or item["strict_connected"]:
            continue
        gap = item["shortest_unknown_bridge_m"]
        gap_text = "n/a" if gap is None else f"{gap:.3f}m"
        print(
            f"{item['pair_id']} {item['side']}: "
            f"class={item['failure_class']} "
            f"new_free={item['new_free_cell_count_in_domain']} "
            f"new_safe={item['new_safe_cell_count_in_domain']} "
            f"max_clearance={item['max_new_free_clearance_m']:.3f}m "
            f"unknown_bridge={gap_text}"
        )


if __name__ == "__main__":
    main()
