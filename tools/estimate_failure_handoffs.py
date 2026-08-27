#!/usr/bin/env python3
"""Estimate authoritative P1-C handoffs at each aisle's first unexpected radius."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate clearance-conditioned handoffs only for aisles with a "
            "first_unexpected_failed_radius_m in aisle geometry diagnostics."
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
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _metadata_from_grid(grid):
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    origin = grid.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("aisle grid.origin must contain [x, y, yaw]")
    return GridMetadata(
        resolution=float(grid["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        width=int(grid["width"]),
        height=int(grid["height"]),
        frame_id=str(grid.get("frame_id", "map")),
    )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.aisle_handoff_boundary import (
        estimate_aisle_handoff_boundary,
    )
    from agt_map_reconstruction.maps.local_blocker_localization import (
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
    metadata = _metadata_from_grid(aisle_payload.get("grid", {}))
    expected_shape = (int(metadata.height), int(metadata.width))
    if base_map.shape != expected_shape:
        raise ValueError(
            f"map shape {base_map.shape} does not match aisle grid {expected_shape}"
        )

    aisle_by_label = {
        str(item["label"]): item
        for item in aisle_payload.get("rectangles", [])
    }
    targets = select_unexpected_failure_targets(diagnostics)

    results = []
    for label, radius in targets.items():
        aisle = aisle_by_label.get(label)
        if aisle is None:
            raise ValueError(f"diagnostic target is missing from aisle bundle: {label}")
        results.append(
            estimate_aisle_handoff_boundary(
                base_map,
                aisle,
                resolution=float(metadata.resolution),
                radius_m=float(radius),
                metadata=metadata,
            )
        )

    payload = {
        "schema_version": 1,
        "source_map": str(map_path),
        "source_aisles": str(aisle_path),
        "source_geometry_diagnostics": str(diagnostics_path),
        "grid": metadata.to_dict(),
        "policy": {
            "radius_selection": "first_unexpected_failed_radius_m per aisle",
            "safe_definition": "free && distance_to_nonfree >= radius",
            "component_selection": (
                "midpoint component; fallback to largest longitudinal span"
            ),
            "map_editing": False,
        },
        "target_count": len(results),
        "handoffs": results,
    }
    (output / "failure_handoffs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("targets:", len(results))
    for item in results:
        if item.get("status") != "ok":
            print(
                f"{item['label']}: radius={item['radius_m']:.2f} "
                f"status={item.get('status')} "
                f"width_eligible={item.get('width_clearance_eligible')}"
            )
            continue
        entry_transition = item.get("entry_transition") or {}
        exit_transition = item.get("exit_transition") or {}
        print(
            f"{item['label']}: radius={item['radius_m']:.2f} "
            f"width_eligible={item['width_clearance_eligible']} "
            f"core_fraction={item['row_core_fraction']:.3f} "
            f"core={item['row_core_start_s_over_l']:.3f}.."
            f"{item['row_core_end_s_over_l']:.3f} "
            f"entry_transition_m={item['entry_transition_length_m']:.2f} "
            f"exit_transition_m={item['exit_transition_length_m']:.2f} "
            f"entry_transition_source={entry_transition.get('dominant_source')} "
            f"exit_transition_source={exit_transition.get('dominant_source')}"
        )


if __name__ == "__main__":
    main()
