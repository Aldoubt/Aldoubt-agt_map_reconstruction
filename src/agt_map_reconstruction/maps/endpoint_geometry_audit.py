"""Audit raw row endpoints against clearance-conditioned handoff geometry.

The existing P1-D3 envelope filters rows with the handoff bundle but fits its
common entry/exit lines from raw row centerline endpoints. This module compares
those raw endpoints with the actual clearance-conditioned handoff cells emitted
by ``aisle_handoff_boundary``. It is diagnostic only and never edits the map.
"""

from __future__ import annotations

import numpy as np


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return value / norm


def _row_axis(rows):
    directions = []
    reference = None
    for row in rows:
        line = np.asarray(row.get("centerline_xy"), dtype=np.float64)
        if line.shape != (2, 2):
            raise ValueError(f"row {row.get('label')} centerline_xy must be 2x2")
        direction = _unit(line[1] - line[0])
        if reference is None:
            reference = direction
        elif float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)
    if not directions:
        raise ValueError("at least one row is required")
    return _unit(np.mean(np.stack(directions), axis=0))


def _fit_line(points, row_axis, cross_axis):
    pts = np.asarray(points, dtype=np.float64)
    u = pts @ row_axis
    v = pts @ cross_axis
    if pts.shape[0] >= 2 and float(np.ptp(v)) > 1e-9:
        slope, intercept = np.polyfit(v, u, 1)
    else:
        slope, intercept = 0.0, float(np.mean(u))
    residual = u - (slope * v + intercept)
    return {
        "slope_du_dv": float(slope),
        "intercept_u": float(intercept),
        "residual_rmse_cells": float(np.sqrt(np.mean(residual * residual))),
    }


def _summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "median_m": None, "p95_m": None, "max_m": None}
    return {
        "count": int(arr.size),
        "median_m": float(np.median(arr)),
        "p95_m": float(np.percentile(arr, 95.0)),
        "max_m": float(np.max(arr)),
    }


def audit_endpoint_geometry(rows, handoffs, *, resolution_m):
    """Compare raw centerline endpoints and real clearance-conditioned handoffs.

    Common entry/exit orientation follows one shared row axis. If a source row
    centerline is reversed relative to that axis, both its raw endpoints and its
    named handoff records are swapped before comparison.
    """
    if float(resolution_m) <= 0.0:
        raise ValueError("resolution_m must be > 0")

    handoff_by_label = {str(item.get("label", "")): item for item in handoffs}
    eligible_rows = []
    for row in rows:
        label = str(row.get("label", ""))
        handoff = handoff_by_label.get(label)
        if handoff is None:
            continue
        if str(handoff.get("status")) != "ok":
            continue
        if handoff.get("width_clearance_eligible") is False:
            continue
        if not handoff.get("entry_handoff") or not handoff.get("exit_handoff"):
            continue
        eligible_rows.append(row)
    if not eligible_rows:
        raise ValueError("no rows have valid clearance-conditioned handoffs")

    row_axis = _row_axis(eligible_rows)
    cross_axis = np.array([-row_axis[1], row_axis[0]], dtype=np.float64)

    records = []
    raw_entry_points = []
    raw_exit_points = []
    handoff_entry_points = []
    handoff_exit_points = []
    polygon_v = []

    for row in eligible_rows:
        label = str(row.get("label", ""))
        handoff = handoff_by_label[label]
        line = np.asarray(row["centerline_xy"], dtype=np.float64)
        forward = float((line[1] - line[0]) @ row_axis) >= 0.0

        raw_entry = line[0].copy() if forward else line[1].copy()
        raw_exit = line[1].copy() if forward else line[0].copy()

        named_entry = np.asarray(handoff["entry_handoff"]["grid_xy"], dtype=np.float64)
        named_exit = np.asarray(handoff["exit_handoff"]["grid_xy"], dtype=np.float64)
        common_entry = named_entry.copy() if forward else named_exit.copy()
        common_exit = named_exit.copy() if forward else named_entry.copy()

        entry_inward_m = float((common_entry - raw_entry) @ row_axis) * float(resolution_m)
        exit_inward_m = float((raw_exit - common_exit) @ row_axis) * float(resolution_m)
        entry_euclid_m = float(np.linalg.norm(common_entry - raw_entry)) * float(resolution_m)
        exit_euclid_m = float(np.linalg.norm(raw_exit - common_exit)) * float(resolution_m)

        records.append(
            {
                "label": label,
                "source_centerline_forward": bool(forward),
                "raw_entry_grid_xy": raw_entry.tolist(),
                "raw_exit_grid_xy": raw_exit.tolist(),
                "handoff_entry_grid_xy": common_entry.tolist(),
                "handoff_exit_grid_xy": common_exit.tolist(),
                "entry_inward_offset_m": entry_inward_m,
                "exit_inward_offset_m": exit_inward_m,
                "entry_euclidean_offset_m": entry_euclid_m,
                "exit_euclidean_offset_m": exit_euclid_m,
                "entry_handoff_clearance_m": handoff["entry_handoff"].get("clearance_m"),
                "exit_handoff_clearance_m": handoff["exit_handoff"].get("clearance_m"),
                "row_core_fraction": handoff.get("row_core_fraction"),
            }
        )
        raw_entry_points.append(raw_entry)
        raw_exit_points.append(raw_exit)
        handoff_entry_points.append(common_entry)
        handoff_exit_points.append(common_exit)
        polygon = np.asarray(row.get("polygon_xy"), dtype=np.float64)
        if polygon.ndim == 2 and polygon.shape[1] == 2:
            polygon_v.extend((polygon @ cross_axis).tolist())

    result = {
        "schema_version": 1,
        "eligible_row_count": len(records),
        "eligible_row_labels": [item["label"] for item in records],
        "row_axis_direction": row_axis.tolist(),
        "cross_row_direction": cross_axis.tolist(),
        "row_cross_span": [float(np.min(polygon_v)), float(np.max(polygon_v))],
        "raw_endpoint_fit": {
            "entry": _fit_line(raw_entry_points, row_axis, cross_axis),
            "exit": _fit_line(raw_exit_points, row_axis, cross_axis),
        },
        "clearance_handoff_fit": {
            "entry": _fit_line(handoff_entry_points, row_axis, cross_axis),
            "exit": _fit_line(handoff_exit_points, row_axis, cross_axis),
        },
        "offset_summary": {
            "entry_inward": _summary([item["entry_inward_offset_m"] for item in records]),
            "exit_inward": _summary([item["exit_inward_offset_m"] for item in records]),
            "entry_euclidean": _summary([item["entry_euclidean_offset_m"] for item in records]),
            "exit_euclidean": _summary([item["exit_euclidean_offset_m"] for item in records]),
        },
        "rows": records,
        "policy": {
            "diagnostic_only": True,
            "d3_geometry_modified": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result
