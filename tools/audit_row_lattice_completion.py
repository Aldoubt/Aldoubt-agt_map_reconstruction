#!/usr/bin/env python3
"""Audit geometry-only row-lattice completion inside wide row-aligned bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Complete expected row-lattice slots inside wide_open_area_candidate "
            "regions without promoting inferred geometry to navigation free space."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--row-band-regions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-observed-slots", type=int, default=4)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _draw_polyline(image, points, color, thickness=1):
    pts = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.polylines(image, [pts], True, color, thickness, lineType=cv2.LINE_AA)


def _draw_centerline(image, points, color, thickness=2):
    pts = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, thickness, lineType=cv2.LINE_AA)


def _label_at_midpoint(image, points, text, color):
    line = np.asarray(points, dtype=np.float64)
    middle = np.mean(line, axis=0)
    x, y = np.rint(middle).astype(int)
    cv2.putText(
        image,
        str(text),
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def _save_display(path, grid_image):
    cv2.imwrite(str(path), np.flipud(grid_image))


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.row_lattice_completion import complete_row_lattice

    map_path = Path(args.map).expanduser().resolve()
    regions_path = Path(args.row_band_regions).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    payload = json.loads(regions_path.read_text(encoding="utf-8"))
    grid = payload["grid"]
    if base.shape != (int(grid["height"]), int(grid["width"])):
        raise ValueError("navigation map and row-band grid shapes differ")

    result = complete_row_lattice(
        payload.get("regions", []),
        resolution_m=float(grid["resolution"]),
        min_observed_slots=int(args.min_observed_slots),
    )
    result["sources"] = {
        "map": str(map_path),
        "row_band_regions": str(regions_path),
    }
    result["source_classification"] = payload.get("classification", {})
    (output / "row_lattice_completion.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    regions = payload.get("regions", [])
    for region in regions:
        if region.get("region_class") != "wide_open_area_candidate":
            continue
        polygon = np.rint(np.asarray(region["polygon_xy"], dtype=float)).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], (160, 80, 160))
    image = cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0)

    for region in regions:
        region_class = region.get("region_class")
        if region_class == "wide_open_area_candidate":
            _draw_polyline(image, region["polygon_xy"], (180, 40, 180), 2)
        elif region_class == "row_aisle":
            _draw_centerline(image, region["centerline_xy"], (0, 180, 0), 1)

    for slot in result.get("slots", []):
        if slot["source"] == "observed_row_aisle":
            color = (0, 210, 0)
            thickness = 2
            text = slot.get("source_band_label") or slot["slot_id"]
        else:
            color = (255, 220, 0)
            thickness = 3
            parent = slot.get("parent_region_label", "")
            text = f"{slot['slot_id']}<{parent}"
        _draw_centerline(image, slot["centerline_xy"], color, thickness)
        _label_at_midpoint(image, slot["centerline_xy"], text, color)

    # Legend in display coordinates: add after vertical flip so text is upright.
    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (370, 88), (30, 30, 30), -1)
    cv2.putText(display, "green: observed row aisle", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1, cv2.LINE_AA)
    cv2.putText(display, "cyan: inferred lattice slot (geometry only)", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(display, "purple: original wide-open candidate", (18, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 40, 180), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "row_lattice_context.png"), display)

    row_count = sum(1 for region in regions if region.get("region_class") == "row_aisle")
    wide_count = sum(1 for region in regions if region.get("region_class") == "wide_open_area_candidate")
    inferred = [slot for slot in result.get("slots", []) if slot["source"] == "lattice_inferred_wide_band"]
    parents = {}
    for slot in inferred:
        parent = slot.get("parent_region_label", "")
        parents[parent] = parents.get(parent, 0) + 1

    print("output:", output)
    print("status:", result["status"])
    print("source_row_aisles:", row_count)
    print("source_wide_open_candidates:", wide_count)
    print("observed_slots:", result.get("observed_slot_count", row_count))
    print("inferred_slots:", result.get("inferred_slot_count", 0))
    if result.get("nominal_pitch_m") is not None:
        print(f"nominal_pitch_m: {result['nominal_pitch_m']:.6f}")
        print(f"nominal_aisle_width_m: {result['nominal_aisle_width_m']:.6f}")
    print("inferred_slots_by_parent:", parents)
    print("automatic_parameter_selection: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
