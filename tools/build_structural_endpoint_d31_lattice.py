#!/usr/bin/env python3
"""Build lattice-aware P1-D3.1 structural endpoints from frozen map evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use row-lattice slots only as geometry for inter-slot ridge search. "
            "Structural evidence still comes exclusively from frozen map HARD cells."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--row-lattice-completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-size-m", type=float, default=0.10)
    parser.add_argument("--min-support-fraction", type=float, default=0.50)
    parser.add_argument("--min-persistence-m", type=float, default=1.00)
    parser.add_argument("--max-internal-gap-m", type=float, default=0.20)
    parser.add_argument("--max-side-endpoint-disagreement-m", type=float, default=0.50)
    parser.add_argument("--residual-floor-m", type=float, default=0.30)
    parser.add_argument("--mad-scale", type=float, default=3.0)
    parser.add_argument("--min-inlier-count", type=int, default=3)
    parser.add_argument("--max-fit-rmse-m", type=float, default=0.50)
    return parser


def _read_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _draw_centerline(image, points, color, thickness=1):
    pts = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, thickness, lineType=cv2.LINE_AA)


def _draw_point(image, point, color, radius=4):
    if point is None:
        return
    x, y = np.rint(np.asarray(point, dtype=np.float64)).astype(int)
    cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_fit(image, fit_payload, row_axis, cross_axis, cross_span, color):
    if fit_payload.get("fit_status") != "ok" or fit_payload.get("fit") is None:
        return
    fit = fit_payload["fit"]
    v = np.asarray(cross_span, dtype=np.float64)
    u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    pts = u[:, None] * row_axis[None, :] + v[:, None] * cross_axis[None, :]
    cv2.line(
        image,
        tuple(np.rint(pts[0]).astype(int)),
        tuple(np.rint(pts[1]).astype(int)),
        color,
        3,
        lineType=cv2.LINE_AA,
    )


def _slot_color(source):
    if source == "lattice_inferred_wide_band":
        return (255, 220, 0)  # cyan-like in BGR
    if source == "observed_split_group":
        return (0, 220, 220)  # yellow
    return (0, 200, 0)  # green


def _status_counts(records, side):
    counts = {}
    for record in records:
        status = str((record.get(side) or {}).get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_lattice_endpoint import (
        build_lattice_structural_endpoint_bundle,
    )

    map_path = Path(args.map).expanduser().resolve()
    lattice_path = Path(args.row_lattice_completion).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    lattice = json.loads(lattice_path.read_text(encoding="utf-8"))
    resolution = lattice.get("sources", {}).get("resolution_m")
    if resolution is None:
        # The lattice bundle is currently paired with the same frozen P1 grid;
        # its slot geometry is in grid cells and the P1 resolution is stored as
        # nominal_pitch_m / nominal_pitch_cells.
        pitch_cells = lattice.get("nominal_pitch_cells")
        pitch_m = lattice.get("nominal_pitch_m")
        if pitch_cells is None or pitch_m is None or float(pitch_cells) <= 0.0:
            raise ValueError("cannot resolve lattice grid resolution")
        resolution = float(pitch_m) / float(pitch_cells)
    resolution = float(resolution)

    result = build_lattice_structural_endpoint_bundle(
        base,
        lattice,
        resolution_m=resolution,
        bin_size_m=float(args.bin_size_m),
        min_support_fraction=float(args.min_support_fraction),
        min_persistence_m=float(args.min_persistence_m),
        max_internal_gap_m=float(args.max_internal_gap_m),
        max_side_endpoint_disagreement_m=float(args.max_side_endpoint_disagreement_m),
        residual_floor_m=float(args.residual_floor_m),
        mad_scale=float(args.mad_scale),
        min_inlier_count=int(args.min_inlier_count),
        max_fit_rmse_m=float(args.max_fit_rmse_m),
    )
    result["sources"] = {
        "map": str(map_path),
        "row_lattice_completion": str(lattice_path),
    }
    result["lattice_fit_provenance"] = {
        "nominal_pitch_m": lattice.get("nominal_pitch_m"),
        "fit_rmse_m": lattice.get("fit_rmse_m"),
        "fit_max_abs_residual_m": lattice.get("fit_max_abs_residual_m"),
        "duplicate_observed_groups": lattice.get("duplicate_observed_groups", []),
    }
    json_path = output / "structural_endpoint_lattice_v1.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for row in result["lattice_rows"]:
        _draw_centerline(image, row["centerline_xy"], _slot_color(row["geometry_source"]), 1)

    fit_rows = {
        side: {item["label"]: item for item in result["robust_boundary"][side]["rows"]}
        for side in ("entry", "exit")
    }
    for record in result["paired_endpoints"]:
        for side in ("entry", "exit"):
            side_record = record[side]
            structural = side_record.get("structural_grid_xy")
            candidate = side_record.get("candidate_grid_xy")
            fit_row = fit_rows[side].get(record["label"], {})
            if structural is not None:
                color = (0, 0, 255) if fit_row.get("inlier") is False else (255, 120, 0)
                _draw_point(image, structural, color, 4)
            elif candidate is not None:
                _draw_point(image, candidate, (0, 165, 255), 4)

    axis = np.asarray(result["row_axis_direction"], dtype=np.float64)
    cross = np.asarray(result["cross_row_direction"], dtype=np.float64)
    cross_values = []
    for row in result["lattice_rows"]:
        cross_values.extend((np.asarray(row["polygon_xy"], dtype=np.float64) @ cross).tolist())
    cross_span = [float(np.min(cross_values)), float(np.max(cross_values))]
    _draw_fit(image, result["robust_boundary"]["entry"], axis, cross, cross_span, (255, 0, 0))
    _draw_fit(image, result["robust_boundary"]["exit"], axis, cross, cross_span, (255, 0, 0))

    display = np.flipud(image).copy()
    cv2.rectangle(display, (8, 8), (470, 116), (30, 30, 30), -1)
    legend = [
        ("green: observed lattice slot", (0, 200, 0)),
        ("yellow: split observed slot", (0, 220, 220)),
        ("cyan: inferred slot; geometry only", (255, 220, 0)),
        ("orange/blue: structural endpoint / common fit", (255, 120, 0)),
    ]
    for index, (text, color) in enumerate(legend):
        cv2.putText(
            display,
            text,
            (18, 30 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "structural_endpoint_context_lattice_v1.png"), display)

    inferred_labels = {
        row["label"]
        for row in result["lattice_rows"]
        if row["geometry_source"] == "lattice_inferred_wide_band"
    }
    inferred_bilateral = {
        side: sum(
            1
            for item in result["paired_endpoints"]
            if item["label"] in inferred_labels and item[side]["status"] == "ok_bilateral"
        )
        for side in ("entry", "exit")
    }

    print("output:", output)
    print("method: lattice_geometry_plus_inter_slot_hard_evidence")
    print("lattice_slots:", result["lattice_slot_count"])
    print("observed_slots:", result["observed_slot_count"])
    print("inferred_slots:", result["inferred_slot_count"])
    print("ridge_profiles:", result["ridge_profile_count"])
    print("ridge_termination_statuses:", {
        status: sum(1 for item in result["ridge_terminations"] if item["status"] == status)
        for status in sorted({item["status"] for item in result["ridge_terminations"]})
    })
    for side in ("entry", "exit"):
        fit = result["robust_boundary"][side]
        print(
            f"{side}: statuses={_status_counts(result['paired_endpoints'], side)} "
            f"inferred_ok_bilateral={inferred_bilateral[side]} "
            f"fit_status={fit['fit_status']} candidates={fit['candidate_count']} "
            f"inliers={fit['inlier_count']} outliers={fit['outlier_count']}"
        )
        if fit.get("fit"):
            print(
                f"  fit_rmse_m={fit['fit']['residual_rmse_m']:.3f} "
                f"gate_m={fit['residual_gate_m']:.3f}"
            )
    print("inferred_slot_supplies_structural_evidence: false")
    print("automatic_parameter_selection: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
