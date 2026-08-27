"""Recover row-aligned aisle rectangles from a corridor-support mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _unit_direction(row_direction):
    direction = np.asarray(row_direction, dtype=float).reshape(-1)
    if direction.size != 2:
        raise ValueError("row_direction must contain exactly two values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("row_direction must be non-zero")
    return direction / norm


def _contiguous_groups(values):
    values = sorted(int(v) for v in values)
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _corners_from_uv(u0, u1, v0, v1, direction, normal):
    return np.asarray([
        u0 * direction + v0 * normal,
        u1 * direction + v0 * normal,
        u1 * direction + v1 * normal,
        u0 * direction + v1 * normal,
    ], dtype=float)


def recover_aisle_rectangles(
    corridor,
    row_direction,
    resolution,
    min_longitudinal_support_ratio=0.50,
    min_width_m=0.30,
    min_length_m=2.0,
):
    """Recover aisle bands using cross-row projection support.

    A true aisle should occupy a substantial fraction of the scene along the
    dominant row direction. Thin headland bridges span many cross-row bins but
    have low longitudinal support, so they are rejected by the support ratio.
    """
    mask = np.asarray(corridor, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("corridor must be a 2D array")
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    if not 0.0 < min_longitudinal_support_ratio <= 1.0:
        raise ValueError("min_longitudinal_support_ratio must be in (0, 1]")

    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return []

    direction = _unit_direction(row_direction)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    points = np.column_stack((xx.astype(float), yy.astype(float)))
    u = points @ direction
    v = points @ normal

    global_u_span_cells = max(float(u.max() - u.min() + 1.0), 1.0)
    v_origin = float(np.floor(v.min()))
    v_bins = np.floor(v - v_origin).astype(int)
    unique_bins, counts = np.unique(v_bins, return_counts=True)
    supported = unique_bins[
        counts / global_u_span_cells >= min_longitudinal_support_ratio
    ]

    aisles = []
    for group in _contiguous_groups(supported):
        group_mask = np.isin(v_bins, group)
        if not np.any(group_mask):
            continue
        u_values = u[group_mask]

        u0 = float(u_values.min() - 0.5)
        u1 = float(u_values.max() + 0.5)
        v0 = float(v_origin + min(group) - 0.5)
        v1 = float(v_origin + max(group) + 0.5)
        width_m = float((v1 - v0) * resolution)
        length_m = float((u1 - u0) * resolution)
        if width_m + 1e-12 < float(min_width_m):
            continue
        if length_m + 1e-12 < float(min_length_m):
            continue

        polygon = _corners_from_uv(u0, u1, v0, v1, direction, normal)
        start = 0.5 * (polygon[0] + polygon[3])
        end = 0.5 * (polygon[1] + polygon[2])
        aisles.append({
            "_cross_row_center": 0.5 * (v0 + v1),
            "polygon_xy": polygon.tolist(),
            "centerline_xy": [start.tolist(), end.tolist()],
            "width_m": width_m,
            "length_m": length_m,
            "heading_rad": float(np.arctan2(direction[1], direction[0])),
        })

    aisles.sort(key=lambda item: item["_cross_row_center"])
    for index, aisle in enumerate(aisles, start=1):
        aisle.pop("_cross_row_center", None)
        aisle["aisle_id"] = index
        aisle["label"] = f"A{index:02d}"
    return aisles


def _grid_points_to_world(points, metadata):
    return [list(metadata.grid_to_world(point[0], point[1])) for point in points]


def write_aisle_bundle(aisles, metadata, path):
    """Persist aisle geometry with legacy grid and explicit map-frame coordinates."""
    rectangles = []
    for aisle in aisles:
        item = dict(aisle)
        item["coordinate_convention"] = "polygon_xy/grid_cell; map fields/metres"
        item["polygon_map_xy_m"] = _grid_points_to_world(
            item["polygon_xy"], metadata
        )
        item["centerline_map_xy_m"] = _grid_points_to_world(
            item["centerline_xy"], metadata
        )
        rectangles.append(item)

    payload = {
        "schema_version": 1,
        "frame_id": metadata.frame_id,
        "grid": metadata.to_dict(),
        "rectangles": rectangles,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return payload
