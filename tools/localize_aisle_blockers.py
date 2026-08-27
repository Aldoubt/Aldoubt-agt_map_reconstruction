#!/usr/bin/env python3
"""Localize geometrically unexpected aisle clearance failures."""

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
            "Reproduce the navigation clearance test for geometrically unexpected "
            "aisle failures and localize their blocking regions."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument("--aisles", required=True, help="aisle_rectangles.json")
    parser.add_argument(
        "--geometry-diagnostics",
        required=True,
        help="aisle_geometry_diagnostics.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-fraction", type=float, default=0.10)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _summary_row(item):
    first = item.get("first_blocker") or {}
    longest = item.get("longest_blocker") or {}
    return {
        "label": item["label"],
        "radius_m": item["radius_m"],
        "validation_pass": item["validation_pass"],
        "failure_region": item.get("failure_region"),
        "dominant_blocking_source": item.get("dominant_blocking_source"),
        "disconnect_mode": item.get("disconnect_mode"),
        "start_probe_safe": item.get("start_probe_safe"),
        "end_probe_safe": item.get("end_probe_safe"),
        "first_blocker_start_s_over_l": first.get("start_s_over_l"),
        "first_blocker_end_s_over_l": first.get("end_s_over_l"),
        "longest_blocker_length_m": longest.get("length_m"),
        "longest_blocker_region": longest.get("region"),
        "longest_blocker_source": longest.get("dominant_blocking_source"),
    }


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.local_blocker_localization import (
        localize_clearance_blocker,
        select_unexpected_failure_targets,
    )

    map_path = Path(args.map).expanduser().resolve()
    aisle_path = Path(args.aisles).expanduser().resolve()
    diagnostics_path = Path(args.geometry_diagnostics).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_map = _read_grid_pgm(map_path)
    aisle_payload = json.loads(aisle_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    grid = aisle_payload.get("grid", {})
    resolution = float(grid["resolution"])
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base_map.shape != expected_shape:
        raise ValueError(
            f"map shape {base_map.shape} does not match aisle grid {expected_shape}"
        )

    aisles = list(aisle_payload.get("rectangles", []))
    aisle_by_label = {str(item["label"]): item for item in aisles}
    targets = select_unexpected_failure_targets(diagnostics)

    results = []
    for label, radius in targets.items():
        if label not in aisle_by_label:
            raise ValueError(f"diagnostic target is missing from aisle bundle: {label}")
        result = localize_clearance_blocker(
            base_map,
            aisle_by_label[label],
            resolution=resolution,
            radius_m=radius,
            probe_fraction=float(args.probe_fraction),
        )
        results.append(result)

    payload = {
        "schema_version": 1,
        "source_map": str(map_path),
        "source_aisles": str(aisle_path),
        "source_geometry_diagnostics": str(diagnostics_path),
        "resolution_m": resolution,
        "probe_fraction": float(args.probe_fraction),
        "target_count": len(results),
        "targets": results,
    }
    (output / "blocker_localization.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = [_summary_row(item) for item in results]
    fieldnames = list(rows[0]) if rows else [
        "label",
        "radius_m",
        "validation_pass",
        "failure_region",
        "dominant_blocking_source",
        "disconnect_mode",
        "start_probe_safe",
        "end_probe_safe",
        "first_blocker_start_s_over_l",
        "first_blocker_end_s_over_l",
        "longest_blocker_length_m",
        "longest_blocker_region",
        "longest_blocker_source",
    ]
    with (output / "blocker_localization.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("output:", output)
    print("targets:", len(results))
    for item in results:
        first = item.get("first_blocker") or {}
        longest = item.get("longest_blocker") or {}
        print(
            f"{item['label']}: radius={item['radius_m']:.2f} "
            f"region={item.get('failure_region')} "
            f"source={item.get('dominant_blocking_source')} "
            f"mode={item.get('disconnect_mode')} "
            f"first={first.get('start_s_over_l')}..{first.get('end_s_over_l')} "
            f"longest_m={longest.get('length_m')}"
        )


if __name__ == "__main__":
    main()
