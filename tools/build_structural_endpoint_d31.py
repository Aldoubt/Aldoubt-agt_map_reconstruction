#!/usr/bin/env python3
"""Build P1-D3.1 structural row-termination assets from frozen map evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Recover bilateral structural row terminations, robustly fit common "
            "entry/exit boundaries, and render them alongside geometric endpoints "
            "and clearance handoffs without modifying the navigation map."
        )
    )
    parser.add_argument("--map", required=True, help="navigation_base_map.pgm")
    parser.add_argument("--row-band-regions", required=True, help="row_band_regions.json")
    parser.add_argument("--handoffs", required=True, help="aisle_handoffs.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strip-width-m", type=float, default=0.40)
    parser.add_argument("--bin-size-m", type=float, default=0.10)
    parser.add_argument("--min-support-fraction", type=float, default=0.50)
    parser.add_argument("--min-persistence-m", type=float, default=1.00)
    parser.add_argument("--max-internal-gap-m", type=float, default=0.20)
    parser.add_argument("--max-side-endpoint-disagreement-m", type=float, default=0.50)
    parser.add_argument("--residual-floor-m", type=float, default=0.30)
    parser.add_argument("--mad-scale", type=float, default=3.0)
    parser.add_argument("--min-inlier-count", type=int, default=3)
    return parser


def _read_grid_pgm(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read PGM: {path}")
    return np.flipud(image).astype(np.uint8, copy=False)


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length row direction")
    return value / norm


def _common_row_axis(rows):
    directions = []
    reference = None
    for row in rows:
        line = np.asarray(row["centerline_xy"], dtype=np.float64)
        direction = _unit(line[1] - line[0])
        if reference is None:
            reference = direction
        elif float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)
    if not directions:
        raise ValueError("no eligible rows")
    return _unit(np.mean(np.stack(directions, axis=0), axis=0))


def _normalize_row_sources(row, handoff, row_axis):
    line = np.asarray(row["centerline_xy"], dtype=np.float64)
    forward = bool(float((line[1] - line[0]) @ row_axis) >= 0.0)
    raw_entry = line[0].copy() if forward else line[1].copy()
    raw_exit = line[1].copy() if forward else line[0].copy()

    named_entry = np.asarray(handoff["entry_handoff"]["grid_xy"], dtype=np.float64)
    named_exit = np.asarray(handoff["exit_handoff"]["grid_xy"], dtype=np.float64)
    handoff_entry = named_entry.copy() if forward else named_exit.copy()
    handoff_exit = named_exit.copy() if forward else named_entry.copy()
    return {
        "source_centerline_forward": forward,
        "raw_geometric_entry_grid_xy": raw_entry.tolist(),
        "raw_geometric_exit_grid_xy": raw_exit.tolist(),
        "clearance_handoff_entry_grid_xy": handoff_entry.tolist(),
        "clearance_handoff_exit_grid_xy": handoff_exit.tolist(),
    }


def _fit_segment(fit, row_axis, cross_axis, v_min, v_max):
    v = np.asarray([float(v_min), float(v_max)], dtype=np.float64)
    u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    points = u[:, None] * row_axis[None, :] + v[:, None] * cross_axis[None, :]
    return np.rint(points).astype(np.int32)


def _draw_point(image, point, color, radius=4):
    if point is None:
        return
    x, y = np.rint(np.asarray(point, dtype=float)).astype(int)
    cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_line(image, a, b, color, thickness=2):
    if a is None or b is None:
        return
    pa = tuple(np.rint(np.asarray(a, dtype=float)).astype(int))
    pb = tuple(np.rint(np.asarray(b, dtype=float)).astype(int))
    cv2.line(image, pa, pb, color, thickness, lineType=cv2.LINE_AA)


def _crop_grid(image_grid, points, pad=45):
    valid = [np.asarray(point, dtype=float) for point in points if point is not None]
    if not valid:
        return None, None
    pts = np.stack(valid, axis=0)
    x0 = max(0, int(np.floor(np.min(pts[:, 0]))) - pad)
    x1 = min(image_grid.shape[1], int(np.ceil(np.max(pts[:, 0]))) + pad + 1)
    y0 = max(0, int(np.floor(np.min(pts[:, 1]))) - pad)
    y1 = min(image_grid.shape[0], int(np.ceil(np.max(pts[:, 1]))) + pad + 1)
    return image_grid[y0:y1, x0:x1].copy(), {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def _draw_legend(display_image):
    x0, y0 = 8, 8
    cv2.rectangle(display_image, (x0, y0), (390, 132), (35, 35, 35), -1)
    entries = [
        ("magenta: geometric aisle endpoint", (255, 0, 255)),
        ("green: clearance handoff", (0, 220, 0)),
        ("cyan: structural endpoint / fit", (255, 220, 0)),
        ("orange: ambiguous structural candidate", (0, 165, 255)),
        ("red: structural-fit outlier", (0, 0, 255)),
    ]
    for index, (text, color) in enumerate(entries):
        y = y0 + 22 + 21 * index
        cv2.circle(display_image, (x0 + 10, y - 5), 4, color, -1)
        cv2.putText(
            display_image,
            text,
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.structural_endpoint_boundary import (
        fit_structural_endpoint_boundaries,
    )
    from agt_map_reconstruction.maps.structural_endpoint_detection import (
        detect_structural_endpoints,
    )
    from agt_map_reconstruction.maps.structural_endpoint_profile import (
        build_structural_support_profile,
    )

    map_path = Path(args.map).expanduser().resolve()
    regions_path = Path(args.row_band_regions).expanduser().resolve()
    handoffs_path = Path(args.handoffs).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = _read_grid_pgm(map_path)
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    handoffs_payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    grid = regions_payload.get("grid", {})
    resolution = float(grid["resolution"])
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if base.shape != expected_shape:
        raise ValueError(f"map shape {base.shape} != region grid {expected_shape}")

    rows_all = [
        item
        for item in regions_payload.get("regions", [])
        if item.get("region_class") == "row_aisle"
    ]
    handoff_by_label = {
        str(item.get("label", "")): item
        for item in handoffs_payload.get("handoffs", [])
    }
    rows = []
    for row in rows_all:
        handoff = handoff_by_label.get(str(row.get("label", "")))
        if handoff is None:
            continue
        if str(handoff.get("status")) != "ok":
            continue
        if handoff.get("width_clearance_eligible") is False:
            continue
        if not handoff.get("entry_handoff") or not handoff.get("exit_handoff"):
            continue
        rows.append(row)
    if not rows:
        raise ValueError("no clearance-width-eligible rows with valid handoffs")

    row_axis = _common_row_axis(rows)
    cross_axis = np.array([-row_axis[1], row_axis[0]], dtype=np.float64)
    polygon_v = []
    for row in rows:
        polygon_v.extend((np.asarray(row["polygon_xy"], dtype=float) @ cross_axis).tolist())
    row_cross_span = [float(np.min(polygon_v)), float(np.max(polygon_v))]

    profile_rows = []
    endpoint_records = []
    boundary_rows = []
    for row in rows:
        label = str(row.get("label", ""))
        handoff = handoff_by_label[label]
        profile = build_structural_support_profile(
            base,
            row,
            resolution_m=resolution,
            strip_width_m=float(args.strip_width_m),
            bin_size_m=float(args.bin_size_m),
            row_axis=row_axis,
        )
        endpoints = detect_structural_endpoints(
            profile,
            min_support_fraction=float(args.min_support_fraction),
            min_persistence_m=float(args.min_persistence_m),
            max_internal_gap_m=float(args.max_internal_gap_m),
            max_side_endpoint_disagreement_m=float(
                args.max_side_endpoint_disagreement_m
            ),
        )
        source_geometry = _normalize_row_sources(row, handoff, row_axis)
        endpoint_record = {
            "label": label,
            "entry": endpoints["entry"],
            "exit": endpoints["exit"],
        }
        endpoint_records.append(endpoint_record)
        profile_rows.append(
            {
                "label": label,
                "profile": profile,
                "endpoint_detection": endpoints,
            }
        )
        boundary_rows.append(
            {
                "label": label,
                **source_geometry,
                "entry": endpoints["entry"],
                "exit": endpoints["exit"],
                "handoff_provenance": {
                    "component_selection": handoff.get("component_selection"),
                    "row_core_fraction": handoff.get("row_core_fraction"),
                    "entry_transition": handoff.get("entry_transition"),
                    "exit_transition": handoff.get("exit_transition"),
                },
            }
        )

    robust = fit_structural_endpoint_boundaries(
        endpoint_records,
        row_axis=row_axis,
        cross_axis=cross_axis,
        resolution_m=resolution,
        residual_floor_m=float(args.residual_floor_m),
        mad_scale=float(args.mad_scale),
        min_inlier_count=int(args.min_inlier_count),
    )

    parameters = {
        "strip_width_m": float(args.strip_width_m),
        "bin_size_m": float(args.bin_size_m),
        "min_support_fraction": float(args.min_support_fraction),
        "min_persistence_m": float(args.min_persistence_m),
        "max_internal_gap_m": float(args.max_internal_gap_m),
        "max_side_endpoint_disagreement_m": float(
            args.max_side_endpoint_disagreement_m
        ),
        "residual_floor_m": float(args.residual_floor_m),
        "mad_scale": float(args.mad_scale),
        "min_inlier_count": int(args.min_inlier_count),
    }
    sources = {
        "map": str(map_path),
        "row_band_regions": str(regions_path),
        "handoffs": str(handoffs_path),
        "structural_support_source": "navigation_base_map HARD cells in bilateral aisle-side strips",
    }
    policy = {
        "geometric_endpoint_is_headland": False,
        "clearance_handoff_is_structural_endpoint": False,
        "unknown_counted_as_structural": False,
        "ambiguous_endpoint_fallback": False,
        "automatic_parameter_selection": False,
        "automatic_acceptance": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }

    profiles_payload = {
        "schema_version": 1,
        "sources": sources,
        "resolution_m": resolution,
        "row_axis_direction": row_axis.tolist(),
        "cross_row_direction": cross_axis.tolist(),
        "row_cross_span": row_cross_span,
        "eligible_row_labels": [str(row.get("label", "")) for row in rows],
        "parameters": parameters,
        "rows": profile_rows,
        "policy": policy,
    }
    boundary_payload = {
        "schema_version": 1,
        "sources": sources,
        "resolution_m": resolution,
        "row_axis_direction": row_axis.tolist(),
        "cross_row_direction": cross_axis.tolist(),
        "row_cross_span": row_cross_span,
        "eligible_row_labels": [str(row.get("label", "")) for row in rows],
        "parameters": parameters,
        "rows": boundary_rows,
        "robust_boundary": robust,
        "policy": policy,
    }

    (output / "structural_endpoint_profiles.json").write_text(
        json.dumps(profiles_payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (output / "structural_endpoint_boundary.json").write_text(
        json.dumps(boundary_payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    gray = np.clip(base.astype(np.int16), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    row_by_label = {str(item.get("label", "")): item for item in rows}
    for label in boundary_payload["eligible_row_labels"]:
        row = row_by_label[label]
        polygon = np.rint(np.asarray(row["polygon_xy"], dtype=float)).astype(np.int32)
        cv2.polylines(image, [polygon], True, (150, 150, 150), 1, lineType=cv2.LINE_AA)
        line = np.asarray(row["centerline_xy"], dtype=float)
        _draw_line(image, line[0], line[1], (170, 170, 170), 1)

    fit_rows = {
        side: {
            item["label"]: item
            for item in robust[side]["rows"]
        }
        for side in ("entry", "exit")
    }
    side_points = {"entry": [], "exit": []}
    for row in boundary_rows:
        for side in ("entry", "exit"):
            raw = row[f"raw_geometric_{side}_grid_xy"]
            handoff = row[f"clearance_handoff_{side}_grid_xy"]
            structural = row[side].get("structural_grid_xy")
            candidate = row[side].get("candidate_grid_xy")
            side_points[side].extend([raw, handoff, structural, candidate])
            _draw_point(image, raw, (255, 0, 255), 3)
            _draw_point(image, handoff, (0, 220, 0), 4)
            fit_row = fit_rows[side].get(row["label"], {})
            if structural is not None:
                color = (0, 0, 255) if fit_row.get("inlier") is False else (255, 220, 0)
                _draw_point(image, structural, color, 4)
            elif candidate is not None:
                _draw_point(image, candidate, (0, 165, 255), 4)

    for side in ("entry", "exit"):
        side_fit = robust[side]
        if side_fit.get("fit_status") != "ok" or side_fit.get("fit") is None:
            continue
        segment = _fit_segment(
            side_fit["fit"],
            row_axis,
            cross_axis,
            row_cross_span[0],
            row_cross_span[1],
        )
        _draw_line(image, segment[0], segment[1], (255, 220, 0), 2)
        side_points[side].extend(segment.tolist())

    display = np.flipud(image).copy()
    _draw_legend(display)
    cv2.imwrite(str(output / "structural_endpoint_context.png"), display)
    for side in ("entry", "exit"):
        crop, bounds = _crop_grid(image, side_points[side])
        boundary_payload[f"{side}_crop_grid_bounds"] = bounds
        if crop is not None:
            cv2.imwrite(
                str(output / f"{side}_structural_endpoint_context.png"),
                np.flipud(crop),
            )

    # Rewrite after adding crop provenance.
    (output / "structural_endpoint_boundary.json").write_text(
        json.dumps(boundary_payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("eligible_rows:", len(rows))
    for side in ("entry", "exit"):
        status_counts = {}
        for row in boundary_rows:
            status = row[side]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        fit = robust[side]
        print(
            f"{side}: statuses={status_counts} "
            f"fit_status={fit['fit_status']} "
            f"candidates={fit['candidate_count']} "
            f"inliers={fit['inlier_count']} "
            f"outliers={fit['outlier_count']}"
        )
        if fit.get("fit") is not None:
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
