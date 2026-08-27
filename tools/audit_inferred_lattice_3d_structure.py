#!/usr/bin/env python3
"""Audit geometry-only inferred lattice ridges with aligned 3D height grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use inferred lattice geometry only to choose target ridge bands; "
            "3D structural evidence comes from aisle-relative low-height relief and "
            "q90-low vertical-extent contrast."
        )
    )
    parser.add_argument("--structural-bundle", required=True)
    parser.add_argument("--low-height", required=True)
    parser.add_argument("--q90-height", required=True)
    parser.add_argument("--point-count", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--aisle-reference-half-width-m", type=float, default=0.20)
    parser.add_argument("--min-topographic-relief-m", type=float, default=0.08)
    parser.add_argument(
        "--min-vertical-extent-m",
        type=float,
        default=0.15,
        help=(
            "minimum ridge-minus-adjacent-aisle q90-low vertical extent contrast; "
            "argument name retained for compatibility"
        ),
    )
    parser.add_argument("--min-support-fraction", type=float, default=0.40)
    parser.add_argument("--min-persistence-m", type=float, default=1.00)
    parser.add_argument("--max-internal-gap-m", type=float, default=0.20)
    parser.add_argument(
        "--min-structural-span-fraction",
        type=float,
        default=0.50,
        help=(
            "minimum fraction of the ridge longitudinal profile spanned between "
            "the detected entry/exit structural terminations; diagnostic quality "
            "gate, not automatic parameter selection"
        ),
    )
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _validate_loaded_grids(low, q90, point_count):
    if low.ndim != 2 or q90.ndim != 2 or point_count.ndim != 2:
        raise ValueError("low_height, q90_height, and point_count must all be 2D")
    if low.shape != q90.shape or low.shape != point_count.shape:
        raise ValueError(
            f"3D grid shapes differ: low={low.shape}, q90={q90.shape}, point_count={point_count.shape}"
        )

    count = np.asarray(point_count, dtype=np.float64)
    if not np.isfinite(count).all() or np.any(count < 0.0):
        raise ValueError("point_count must contain finite non-negative values")
    if not np.allclose(count, np.rint(count), atol=1e-6, rtol=0.0):
        raise ValueError(
            "point_count must be integer-valued; q90_height and point_count may have been swapped"
        )

    low_f = np.asarray(low, dtype=np.float64)
    q90_f = np.asarray(q90, dtype=np.float64)
    finite = np.isfinite(low_f) & np.isfinite(q90_f)
    if np.any(q90_f[finite] + 1e-6 < low_f[finite]):
        raise ValueError(
            "q90_height is below low_height on finite cells; check the height-grid inputs"
        )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.inferred_lattice_3d_structure import (
        audit_inferred_lattice_3d_structure,
    )

    source_path = Path(args.structural_bundle).expanduser().resolve()
    low_path = Path(args.low_height).expanduser().resolve()
    q90_path = Path(args.q90_height).expanduser().resolve()
    count_path = Path(args.point_count).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(source_path.read_text(encoding="utf-8"))
    low = np.load(low_path)
    q90 = np.load(q90_path)
    point_count = np.load(count_path)
    _validate_loaded_grids(low, q90, point_count)

    map_path_text = (bundle.get("sources") or {}).get("map")
    if map_path_text:
        base = _read_pgm(Path(map_path_text))
        if base.shape != low.shape:
            raise ValueError(
                f"3D grids shape {low.shape} does not match frozen navigation map shape {base.shape}"
            )
    else:
        base = np.full(low.shape, 205, dtype=np.uint8)

    result = audit_inferred_lattice_3d_structure(
        bundle,
        low,
        q90,
        point_count,
        min_points_per_cell=int(args.min_points_per_cell),
        aisle_reference_half_width_m=float(args.aisle_reference_half_width_m),
        min_topographic_relief_m=float(args.min_topographic_relief_m),
        min_vertical_extent_m=float(args.min_vertical_extent_m),
        min_support_fraction=float(args.min_support_fraction),
        min_persistence_m=float(args.min_persistence_m),
        max_internal_gap_m=float(args.max_internal_gap_m),
        min_structural_span_fraction=float(args.min_structural_span_fraction),
    )
    result["sources"] = {
        "structural_bundle": str(source_path),
        "low_height": str(low_path),
        "q90_height": str(q90_path),
        "point_count": str(count_path),
        "map": map_path_text,
    }
    json_path = output / "inferred_lattice_3d_structure.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    profiles = {str(item["ridge_id"]): item for item in bundle.get("ridge_profiles") or []}
    for audit in result["ridge_audits"]:
        profile = profiles[audit["ridge_id"]]
        centers = np.asarray(profile.get("bin_center_grid_xy"), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[0] < 2:
            continue
        if audit["status"] == "ok_3d_structural_support":
            color = (0, 200, 0)
        elif audit["status"] == "insufficient_longitudinal_structural_span":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.polylines(
            image,
            [np.rint(centers).astype(np.int32)],
            False,
            color,
            3,
            lineType=cv2.LINE_AA,
        )
        for side in ("entry", "exit"):
            point = audit.get(f"{side}_grid_xy")
            if point is None:
                continue
            x, y = np.rint(np.asarray(point, dtype=np.float64)).astype(int)
            cv2.circle(image, (x, y), 5, color, -1, lineType=cv2.LINE_AA)

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (700, 116), (30, 30, 30), -1)
    legend = [
        ("green: inferred-adjacent ridge with endpoint-eligible aisle-relative 3D support", (0, 200, 0)),
        ("orange: local 3D structure exists but longitudinal span is insufficient", (0, 165, 255)),
        ("red: inferred-adjacent ridge lacks sustained aisle-relative 3D structure", (0, 0, 255)),
        ("lattice geometry selects targets only; it never supplies 3D evidence", (255, 220, 0)),
    ]
    for index, (text, color) in enumerate(legend):
        cv2.putText(
            display,
            text,
            (18, 30 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "inferred_lattice_3d_structure.png"), display)

    print("output:", output)
    print("method:", result["method"])
    print("target_ridges:", result["target_ridge_count"])
    print("supported_target_ridges:", result["supported_target_ridge_count"])
    print("unsupported_target_ridges:", result["unsupported_target_ridge_count"])
    for item in result["ridge_audits"]:
        summary = item["evidence_summary"]
        span = item.get("structural_span_fraction")
        span_text = "none" if span is None else f"{span:.6f}"
        print(
            f"{item['ridge_id']}: status={item['status']} "
            f"supported_bins={summary['supported_bin_count']} "
            f"supported_bin_fraction={summary['supported_bin_fraction']:.6f} "
            f"structural_span_fraction={span_text} "
            f"topographic_bins={summary['topographic_supported_bin_count']} "
            f"vertical_contrast_bins={summary['vertical_supported_bin_count']} "
            f"valid_cells={summary['valid_cell_count']}"
        )
    print("vertical_extent_cue_is_aisle_relative: true")
    print("local_3d_structure_does_not_imply_full_ridge_endpoint_support: true")
    print("inferred_slot_supplies_3d_evidence: false")
    print("automatic_parameter_selection: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
