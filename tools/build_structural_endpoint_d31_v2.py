#!/usr/bin/env python3
"""Build P1-D3.1 v2 structural endpoints from inter-aisle ridge evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Recover row terminations from inter-aisle ridge bands. This v2 path "
            "does not use generic outer-wall HARD strips or longest internal runs."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--row-band-regions", required=True)
    parser.add_argument("--handoffs", required=True)
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


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    return value / np.linalg.norm(value)


def _common_axis(rows):
    directions = []
    ref = None
    for row in rows:
        line = np.asarray(row["centerline_xy"], dtype=np.float64)
        d = _unit(line[1] - line[0])
        if ref is None:
            ref = d
        elif float(d @ ref) < 0.0:
            d = -d
        directions.append(d)
    return _unit(np.mean(np.stack(directions), axis=0))


def _oriented_raw(row, row_axis):
    line = np.asarray(row["centerline_xy"], dtype=np.float64)
    return (line[0], line[1]) if float((line[1] - line[0]) @ row_axis) >= 0.0 else (line[1], line[0])


def _oriented_handoff(handoff, row_forward):
    entry = np.asarray(handoff["entry_handoff"]["grid_xy"], dtype=np.float64)
    exit_ = np.asarray(handoff["exit_handoff"]["grid_xy"], dtype=np.float64)
    return (entry, exit_) if row_forward else (exit_, entry)


def _draw_point(image, point, color, radius=4):
    if point is None:
        return
    x, y = np.rint(np.asarray(point, dtype=float)).astype(int)
    cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_fit(image, fit_payload, row_axis, cross_axis, cross_span, color):
    if fit_payload.get("fit_status") != "ok" or fit_payload.get("fit") is None:
        return
    fit = fit_payload["fit"]
    v = np.asarray(cross_span, dtype=np.float64)
    u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    pts = u[:, None] * row_axis[None, :] + v[:, None] * cross_axis[None, :]
    p0 = tuple(np.rint(pts[0]).astype(int))
    p1 = tuple(np.rint(pts[1]).astype(int))
    cv2.line(image, p0, p1, color, 2, lineType=cv2.LINE_AA)


def _save_display(path, grid_image):
    cv2.imwrite(str(path), np.flipud(grid_image))


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_boundary import (
        fit_structural_endpoint_boundaries,
    )
    from agt_map_reconstruction.maps.structural_ridge_endpoint import (
        build_inter_aisle_ridge_profiles,
        detect_ridge_terminations,
        pair_aisle_structural_endpoints,
    )

    map_path = Path(args.map).expanduser().resolve()
    regions_path = Path(args.row_band_regions).expanduser().resolve()
    handoffs_path = Path(args.handoffs).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_pgm(map_path)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    grid = regions_payload["grid"]
    resolution = float(grid["resolution"])
    if base.shape != (int(grid["height"]), int(grid["width"])):
        raise ValueError("navigation map and row region grid shapes differ")

    all_rows = [r for r in regions_payload.get("regions", []) if r.get("region_class") == "row_aisle"]
    if len(all_rows) < 3:
        raise ValueError("D3.1 v2 requires at least three recovered row aisles")
    row_axis = _common_axis(all_rows)
    cross_axis = np.array([-row_axis[1], row_axis[0]], dtype=np.float64)

    handoff_by_label = {str(h["label"]): h for h in handoffs_payload.get("handoffs", [])}
    eligible_labels = []
    for row in all_rows:
        label = str(row["label"])
        handoff = handoff_by_label.get(label)
        if not handoff or handoff.get("status") != "ok":
            continue
        if handoff.get("width_clearance_eligible") is False:
            continue
        if handoff.get("entry_handoff") and handoff.get("exit_handoff"):
            eligible_labels.append(label)

    profiles = build_inter_aisle_ridge_profiles(
        base,
        all_rows,
        resolution_m=resolution,
        bin_size_m=float(args.bin_size_m),
        row_axis=row_axis,
    )
    ridge_terminations = [
        detect_ridge_terminations(
            profile,
            min_support_fraction=float(args.min_support_fraction),
            min_persistence_m=float(args.min_persistence_m),
            max_internal_gap_m=float(args.max_internal_gap_m),
        )
        for profile in profiles
    ]
    paired_all = pair_aisle_structural_endpoints(
        all_rows,
        ridge_terminations,
        row_axis=row_axis,
        max_side_endpoint_disagreement_m=float(args.max_side_endpoint_disagreement_m),
    )
    paired_by_label = {item["label"]: item for item in paired_all}
    endpoint_records = [paired_by_label[label] for label in eligible_labels if label in paired_by_label]

    robust = fit_structural_endpoint_boundaries(
        endpoint_records,
        row_axis=row_axis,
        cross_axis=cross_axis,
        resolution_m=resolution,
        residual_floor_m=float(args.residual_floor_m),
        mad_scale=float(args.mad_scale),
        min_inlier_count=int(args.min_inlier_count),
        max_fit_rmse_m=float(args.max_fit_rmse_m),
    )

    row_by_label = {str(row["label"]): row for row in all_rows}
    rows_out = []
    cross_values = []
    for label in eligible_labels:
        row = row_by_label[label]
        handoff = handoff_by_label[label]
        pair = paired_by_label[label]
        raw_entry, raw_exit = _oriented_raw(row, row_axis)
        source_forward = bool(float((np.asarray(row["centerline_xy"])[1] - np.asarray(row["centerline_xy"])[0]) @ row_axis) >= 0.0)
        handoff_entry, handoff_exit = _oriented_handoff(handoff, source_forward)
        polygon = np.asarray(row["polygon_xy"], dtype=np.float64)
        cross_values.extend((polygon @ cross_axis).tolist())
        rows_out.append(
            {
                "label": label,
                "raw_geometric_entry_grid_xy": raw_entry.tolist(),
                "raw_geometric_exit_grid_xy": raw_exit.tolist(),
                "clearance_handoff_entry_grid_xy": handoff_entry.tolist(),
                "clearance_handoff_exit_grid_xy": handoff_exit.tolist(),
                "left_ridge_id": pair.get("left_ridge_id"),
                "right_ridge_id": pair.get("right_ridge_id"),
                "entry": pair["entry"],
                "exit": pair["exit"],
            }
        )
    cross_span = [float(np.min(cross_values)), float(np.max(cross_values))]

    payload = {
        "schema_version": 2,
        "method": "inter_aisle_ridge_boundary_anchored",
        "sources": {
            "map": str(map_path),
            "row_band_regions": str(regions_path),
            "handoffs": str(handoffs_path),
        },
        "resolution_m": resolution,
        "row_axis_direction": row_axis.tolist(),
        "cross_row_direction": cross_axis.tolist(),
        "row_cross_span": cross_span,
        "eligible_row_labels": eligible_labels,
        "parameters": {
            "bin_size_m": float(args.bin_size_m),
            "min_support_fraction": float(args.min_support_fraction),
            "min_persistence_m": float(args.min_persistence_m),
            "max_internal_gap_m": float(args.max_internal_gap_m),
            "max_side_endpoint_disagreement_m": float(args.max_side_endpoint_disagreement_m),
            "residual_floor_m": float(args.residual_floor_m),
            "mad_scale": float(args.mad_scale),
            "min_inlier_count": int(args.min_inlier_count),
            "max_fit_rmse_m": float(args.max_fit_rmse_m),
        },
        "ridge_profiles": profiles,
        "ridge_terminations": ridge_terminations,
        "rows": rows_out,
        "robust_boundary": robust,
        "policy": {
            "generic_outer_wall_used_as_ridge": False,
            "longest_internal_run_used": False,
            "poor_fit_promoted": False,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    (output / "structural_endpoint_boundary_v2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for row in all_rows:
        polygon = np.rint(np.asarray(row["polygon_xy"], dtype=float)).astype(np.int32)
        cv2.polylines(image, [polygon], True, (170, 170, 170), 1, lineType=cv2.LINE_AA)
    for profile in profiles:
        v0, v1 = profile["ridge_cross_span_cells"]
        edges = profile["bin_edges_u_cells"]
        corners = np.asarray([
            float(edges[0]) * row_axis + float(v0) * cross_axis,
            float(edges[-1]) * row_axis + float(v0) * cross_axis,
            float(edges[-1]) * row_axis + float(v1) * cross_axis,
            float(edges[0]) * row_axis + float(v1) * cross_axis,
        ])
        cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (120, 90, 20), 1, lineType=cv2.LINE_AA)

    fit_rows = {
        side: {item["label"]: item for item in robust[side]["rows"]}
        for side in ("entry", "exit")
    }
    for row in rows_out:
        for side in ("entry", "exit"):
            _draw_point(image, row[f"raw_geometric_{side}_grid_xy"], (255, 0, 255), 3)
            _draw_point(image, row[f"clearance_handoff_{side}_grid_xy"], (0, 220, 0), 3)
            record = row[side]
            structural = record.get("structural_grid_xy")
            candidate = record.get("candidate_grid_xy")
            fit_row = fit_rows[side].get(row["label"], {})
            if structural is not None:
                color = (0, 0, 255) if fit_row.get("inlier") is False else (255, 220, 0)
                _draw_point(image, structural, color, 4)
            elif candidate is not None:
                _draw_point(image, candidate, (0, 165, 255), 4)

    _draw_fit(image, robust["entry"], row_axis, cross_axis, cross_span, (255, 220, 0))
    _draw_fit(image, robust["exit"], row_axis, cross_axis, cross_span, (255, 220, 0))
    _save_display(output / "structural_endpoint_context_v2.png", image)

    print("output:", output)
    print("method: inter_aisle_ridge_boundary_anchored")
    print("ridge_profiles:", len(profiles))
    for side in ("entry", "exit"):
        statuses = {}
        for row in endpoint_records:
            status = row[side]["status"]
            statuses[status] = statuses.get(status, 0) + 1
        fit = robust[side]
        print(
            f"{side}: statuses={statuses} fit_status={fit['fit_status']} "
            f"candidates={fit['candidate_count']} inliers={fit['inlier_count']} "
            f"outliers={fit['outlier_count']}"
        )
        if fit.get("fit"):
            print(
                f"  fit_rmse_m={fit['fit']['residual_rmse_m']:.3f} "
                f"gate_m={fit['residual_gate_m']:.3f}"
            )
    print("automatic_parameter_selection: false")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
