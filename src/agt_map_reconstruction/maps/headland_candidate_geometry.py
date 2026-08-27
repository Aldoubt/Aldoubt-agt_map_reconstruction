"""Geometric audit for wide row-band candidates before any headland promotion."""

from __future__ import annotations

import numpy as np

from .navigation_export import rasterize_aisles


def _unit(vector):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length direction")
    return value / norm


def _row_axis(row_aisles):
    directions = []
    reference = None
    for aisle in row_aisles:
        line = np.asarray(aisle.get("centerline_xy"), dtype=float)
        if line.shape != (2, 2):
            raise ValueError(f"row aisle {aisle.get('label')} centerline_xy must be 2x2")
        direction = _unit(line[1] - line[0])
        if reference is None:
            reference = direction
        elif float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)
    if not directions:
        raise ValueError("at least one row_aisle is required")
    return _unit(np.mean(np.stack(directions, axis=0), axis=0))


def _oriented_endpoints(aisle, row_axis):
    line = np.asarray(aisle["centerline_xy"], dtype=float)
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
    return {
        "slope_du_dv": float(slope),
        "intercept_u": float(intercept),
        "residual_rmse_cells": float(
            np.sqrt(np.mean((u - (slope * v + intercept)) ** 2))
        ),
    }


def _long_axis_alignment(region, row_axis):
    centerline = region.get("centerline_xy")
    if centerline is not None:
        line = np.asarray(centerline, dtype=float)
        if line.shape == (2, 2) and float(np.linalg.norm(line[1] - line[0])) > 1e-12:
            return abs(float(_unit(line[1] - line[0]) @ row_axis))

    polygon = np.asarray(region.get("polygon_xy"), dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 2 or polygon.shape[1] != 2:
        raise ValueError(f"candidate {region.get('label')} needs polygon_xy")
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    return abs(float(_unit(edge) @ row_axis))


def _interval_overlap_fraction(a0, a1, b0, b1):
    left = max(float(a0), float(b0))
    right = min(float(a1), float(b1))
    overlap = max(0.0, right - left)
    denom = max(1e-12, float(b1) - float(b0))
    return float(np.clip(overlap / denom, 0.0, 1.0))


def analyze_headland_candidate_geometry(regions, grid_shape):
    """Audit whether wide-band candidates geometrically resemble a headland.

    This function deliberately emits continuous descriptors rather than a binary
    HEADLAND label. A genuine row-end headland should usually cover much of the
    cross-row span and lie primarily beyond either the common entry or exit
    endpoint line. A row-parallel exterior band instead tends to align with the
    row axis and have little cross-row overlap.
    """
    rows = [
        dict(item)
        for item in regions
        if str(item.get("region_class", "")) == "row_aisle"
    ]
    candidates = [
        dict(item)
        for item in regions
        if str(item.get("region_class", "")) == "wide_open_area_candidate"
    ]
    if len(grid_shape) != 2 or int(grid_shape[0]) <= 0 or int(grid_shape[1]) <= 0:
        raise ValueError("grid_shape must be (height, width)")
    shape = (int(grid_shape[0]), int(grid_shape[1]))

    row_axis = _row_axis(rows)
    cross_axis = np.array([-row_axis[1], row_axis[0]], dtype=float)

    starts = []
    ends = []
    center_v = []
    for aisle in rows:
        start, end = _oriented_endpoints(aisle, row_axis)
        starts.append(start)
        ends.append(end)
        center_v.append(float((0.5 * (start + end)) @ cross_axis))

    entry_fit = _fit_endpoint_line(starts, row_axis, cross_axis)
    exit_fit = _fit_endpoint_line(ends, row_axis, cross_axis)
    row_v_min = float(np.min(center_v))
    row_v_max = float(np.max(center_v))

    audited = []
    for region in candidates:
        mask = rasterize_aisles([region], shape)
        yy, xx = np.nonzero(mask)
        if yy.size == 0:
            raise ValueError(f"candidate {region.get('label')} rasterizes to no cells")
        points = np.column_stack((xx.astype(float), yy.astype(float)))
        u = points @ row_axis
        v = points @ cross_axis

        entry_boundary_u = (
            entry_fit["slope_du_dv"] * v + entry_fit["intercept_u"]
        )
        exit_boundary_u = (
            exit_fit["slope_du_dv"] * v + exit_fit["intercept_u"]
        )
        entry_outward = u < entry_boundary_u - 1e-12
        exit_outward = u > exit_boundary_u + 1e-12

        candidate_v_min = float(np.min(v))
        candidate_v_max = float(np.max(v))
        cross_overlap = _interval_overlap_fraction(
            candidate_v_min,
            candidate_v_max,
            row_v_min,
            row_v_max,
        )

        audited.append({
            "label": str(region.get("label", "")),
            "region_class": "wide_open_area_candidate",
            "source_band_label": region.get("source_band_label"),
            "row_axis_alignment": float(_long_axis_alignment(region, row_axis)),
            "cross_row_overlap_fraction": cross_overlap,
            "entry_outward_fraction": float(np.mean(entry_outward)),
            "exit_outward_fraction": float(np.mean(exit_outward)),
            "candidate_cross_row_min": candidate_v_min,
            "candidate_cross_row_max": candidate_v_max,
            "semantic_promotion": False,
        })

    return {
        "schema_version": 1,
        "row_aisle_count": len(rows),
        "candidate_count": len(candidates),
        "row_axis_direction": [float(row_axis[0]), float(row_axis[1])],
        "cross_row_direction": [float(cross_axis[0]), float(cross_axis[1])],
        "row_cross_span": [row_v_min, row_v_max],
        "entry_endpoint_fit": entry_fit,
        "exit_endpoint_fit": exit_fit,
        "interpretation": {
            "row_axis_alignment": "1=row-parallel, 0=cross-row",
            "cross_row_overlap_fraction": "fraction of row cross-span covered by candidate",
            "entry_outward_fraction": "candidate-cell fraction beyond common entry endpoint line",
            "exit_outward_fraction": "candidate-cell fraction beyond common exit endpoint line",
            "semantic_promotion": False,
        },
        "candidates": audited,
    }
