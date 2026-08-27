"""Recover evidence envelopes outside common row endpoints without semantic promotion."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _unit(vector):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length direction")
    return value / norm


def _row_axis(rows):
    directions = []
    reference = None
    for row in rows:
        line = np.asarray(row.get("centerline_xy"), dtype=float)
        if line.shape != (2, 2):
            raise ValueError(f"row {row.get('label')} centerline_xy must be 2x2")
        direction = _unit(line[1] - line[0])
        if reference is None:
            reference = direction
        elif float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)
    if not directions:
        raise ValueError("at least one eligible row is required")
    return _unit(np.mean(np.stack(directions, axis=0), axis=0))


def _oriented_endpoints(row, row_axis):
    line = np.asarray(row["centerline_xy"], dtype=float)
    start, end = line[0].copy(), line[1].copy()
    if float((end - start) @ row_axis) < 0.0:
        start, end = end, start
    return start, end


def _fit_endpoint_line(points, row_axis, cross_axis):
    points = np.asarray(points, dtype=float)
    u = points @ row_axis
    v = points @ cross_axis
    if points.shape[0] >= 2 and float(np.ptp(v)) > 1e-9:
        slope, intercept = np.polyfit(v, u, 1)
    else:
        slope, intercept = 0.0, float(np.mean(u))
    residual = u - (slope * v + intercept)
    return {
        "slope_du_dv": float(slope),
        "intercept_u": float(intercept),
        "residual_rmse_cells": float(np.sqrt(np.mean(residual * residual))),
    }


def _safe_masks(base, resolution, radius_m):
    free = base == FREE_VALUE
    hard = base == OCCUPIED_VALUE
    strict_distance = ndimage.distance_transform_edt(free) * float(resolution)
    strict = free & (strict_distance + 1e-12 >= float(radius_m))

    nonhard = ~hard
    hard_distance = ndimage.distance_transform_edt(nonhard) * float(resolution)
    relaxed = nonhard & (hard_distance + 1e-12 >= float(radius_m))
    return strict, relaxed


def _component_record(
    component,
    points_xy,
    u,
    v,
    boundary_u,
    side,
    row_v_min,
    row_v_max,
    endpoint_points,
    base,
    resolution,
):
    local_points = points_xy[component]
    local_u = u[component]
    local_v = v[component]
    local_boundary = boundary_u[component]
    if side == "entry":
        depth_cells = local_boundary - local_u
    else:
        depth_cells = local_u - local_boundary

    comp_v_min = float(np.min(local_v))
    comp_v_max = float(np.max(local_v))
    overlap = max(0.0, min(comp_v_max, row_v_max) - max(comp_v_min, row_v_min))
    row_span = max(1e-12, float(row_v_max) - float(row_v_min))

    mask = np.zeros(base.shape, dtype=bool)
    x = local_points[:, 0].astype(int)
    y = local_points[:, 1].astype(int)
    mask[y, x] = True
    distance_to_component = ndimage.distance_transform_edt(~mask) * float(resolution)
    endpoint_distances = []
    for point in endpoint_points:
        px, py = np.rint(point).astype(int)
        if 0 <= py < base.shape[0] and 0 <= px < base.shape[1]:
            endpoint_distances.append(float(distance_to_component[py, px]))
        else:
            endpoint_distances.append(float("inf"))

    local_unknown = base[y, x] == UNKNOWN_VALUE
    finite_endpoint_distances = [d for d in endpoint_distances if np.isfinite(d)]
    return {
        "cell_count": int(local_points.shape[0]),
        "area_m2": float(local_points.shape[0]) * float(resolution) ** 2,
        "cross_row_min": comp_v_min,
        "cross_row_max": comp_v_max,
        "cross_row_coverage_fraction": float(np.clip(overlap / row_span, 0.0, 1.0)),
        "max_outward_depth_m": float(np.max(depth_cells)) * float(resolution),
        "median_outward_depth_m": float(np.median(depth_cells)) * float(resolution),
        "unknown_cell_fraction": float(np.mean(local_unknown)),
        "endpoint_distance_m": endpoint_distances,
        "endpoint_distance_median_m": (
            float(np.median(finite_endpoint_distances))
            if finite_endpoint_distances else None
        ),
        "endpoint_distance_max_m": (
            float(np.max(finite_endpoint_distances))
            if finite_endpoint_distances else None
        ),
    }


def _summarize_policy(
    safe,
    roi,
    points_xy,
    u,
    v,
    boundary_u,
    side,
    row_v_min,
    row_v_max,
    endpoint_points,
    base,
    resolution,
):
    labels, count = ndimage.label(safe & roi)
    records = []
    for component_id in range(1, int(count) + 1):
        component = labels[points_xy[:, 1].astype(int), points_xy[:, 0].astype(int)] == component_id
        if not np.any(component):
            continue
        record = _component_record(
            component,
            points_xy,
            u,
            v,
            boundary_u,
            side,
            row_v_min,
            row_v_max,
            endpoint_points,
            base,
            resolution,
        )
        record["component_id"] = int(component_id)
        records.append(record)

    best = max(
        records,
        key=lambda item: (
            item["cross_row_coverage_fraction"],
            -float("inf") if item["endpoint_distance_median_m"] is None else -item["endpoint_distance_median_m"],
            item["max_outward_depth_m"],
            item["cell_count"],
        ),
        default=None,
    )
    return {
        "component_count": len(records),
        "best_component": best,
        "components": records,
    }


def analyze_endpoint_side_envelopes(
    base_map,
    row_aisles,
    handoffs,
    resolution,
    radius_m,
):
    """Analyze strict and unknown-relaxed evidence outside common row endpoints.

    Rows that are not clearance-width eligible in the handoff bundle are excluded.
    The result intentionally stays descriptive: it reports continuous evidence
    envelopes for the common entry and exit sides and never promotes a region to
    HEADLAND automatically.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    if float(resolution) <= 0.0:
        raise ValueError("resolution must be > 0")
    if float(radius_m) < 0.0:
        raise ValueError("radius_m must be >= 0")

    handoff_by_label = {str(item.get("label")): item for item in handoffs}
    eligible_rows = []
    for row in row_aisles:
        label = str(row.get("label", ""))
        handoff = handoff_by_label.get(label)
        if handoff is None:
            continue
        if str(handoff.get("status")) != "ok":
            continue
        if handoff.get("width_clearance_eligible") is False:
            continue
        eligible_rows.append(dict(row))
    if not eligible_rows:
        raise ValueError("no clearance-width eligible rows available")

    row_axis = _row_axis(eligible_rows)
    cross_axis = np.array([-row_axis[1], row_axis[0]], dtype=float)

    entry_points = []
    exit_points = []
    polygon_v = []
    eligible_labels = []
    for row in eligible_rows:
        entry, exit_ = _oriented_endpoints(row, row_axis)
        entry_points.append(entry)
        exit_points.append(exit_)
        polygon = np.asarray(row.get("polygon_xy"), dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"row {row.get('label')} polygon_xy must be Nx2")
        polygon_v.extend((polygon @ cross_axis).tolist())
        eligible_labels.append(str(row.get("label", "")))

    entry_fit = _fit_endpoint_line(entry_points, row_axis, cross_axis)
    exit_fit = _fit_endpoint_line(exit_points, row_axis, cross_axis)
    row_v_min = float(np.min(polygon_v))
    row_v_max = float(np.max(polygon_v))

    yy, xx = np.indices(base.shape)
    points_xy = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(float)
    u = points_xy @ row_axis
    v = points_xy @ cross_axis
    cross_roi = (v >= row_v_min - 1e-12) & (v <= row_v_max + 1e-12)

    strict_safe, relaxed_safe = _safe_masks(base, resolution, radius_m)
    sides = {}
    for side, endpoint_points, fit in (
        ("entry", entry_points, entry_fit),
        ("exit", exit_points, exit_fit),
    ):
        boundary_u = fit["slope_du_dv"] * v + fit["intercept_u"]
        if side == "entry":
            outward = u < boundary_u - 1e-12
        else:
            outward = u > boundary_u + 1e-12
        roi_flat = cross_roi & outward
        roi = roi_flat.reshape(base.shape)
        sides[side] = {
            "endpoint_fit": fit,
            "roi_cell_count": int(np.count_nonzero(roi)),
            "strict": _summarize_policy(
                strict_safe,
                roi,
                points_xy,
                u,
                v,
                boundary_u,
                side,
                row_v_min,
                row_v_max,
                endpoint_points,
                base,
                resolution,
            ),
            "relaxed_unknown_allowed": _summarize_policy(
                relaxed_safe,
                roi,
                points_xy,
                u,
                v,
                boundary_u,
                side,
                row_v_min,
                row_v_max,
                endpoint_points,
                base,
                resolution,
            ),
        }

    return {
        "schema_version": 1,
        "radius_m": float(radius_m),
        "eligible_row_count": len(eligible_rows),
        "eligible_row_labels": eligible_labels,
        "row_axis_direction": [float(row_axis[0]), float(row_axis[1])],
        "cross_row_direction": [float(cross_axis[0]), float(cross_axis[1])],
        "row_cross_span": [row_v_min, row_v_max],
        "policy": {
            "strict": "confirmed free with requested clearance",
            "relaxed_unknown_allowed": "unknown allowed; hard occupied still blocks",
            "semantic_promotion": False,
        },
        "sides": sides,
    }
