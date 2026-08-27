"""Estimate clearance-conditioned row-core and aisle handoff boundaries."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    rasterize_aisles,
)


def _point_label(labels, xy):
    x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _distance_to_mask(mask, resolution):
    mask = np.asarray(mask, dtype=bool)
    if np.any(mask):
        return ndimage.distance_transform_edt(~mask) * float(resolution)
    return np.full(mask.shape, np.inf, dtype=float)


def _boundary_source(point_xy, distance_hard, distance_unknown, resolution):
    x, y = np.rint(np.asarray(point_xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= distance_hard.shape[0] or x >= distance_hard.shape[1]:
        return "out_of_map"
    dh = float(distance_hard[y, x])
    du = float(distance_unknown[y, x])
    if not np.isfinite(dh) and not np.isfinite(du):
        return "none"
    if not np.isfinite(dh):
        return "unknown"
    if not np.isfinite(du):
        return "hard"
    tolerance = 0.25 * float(resolution)
    if abs(dh - du) <= tolerance:
        return "mixed"
    return "hard" if dh < du else "unknown"


def _interpolate_map_centerline(aisle, t):
    map_line = aisle.get("centerline_map_xy_m")
    if map_line is None:
        return None
    points = np.asarray(map_line, dtype=float)
    if points.shape != (2, 2):
        raise ValueError("centerline_map_xy_m must be 2x2 when present")
    point = points[0] + float(t) * (points[1] - points[0])
    return [float(point[0]), float(point[1])]


def _handoff_record(
    aisle,
    t,
    full_start,
    vector,
    length_m,
    distance_hard,
    distance_unknown,
    resolution,
):
    point = full_start + float(t) * vector
    heading = float(
        aisle.get("heading_rad", np.arctan2(vector[1], vector[0]))
    )
    return {
        "s_over_l": float(t),
        "s_m": float(t) * float(length_m),
        "grid_xy": [float(point[0]), float(point[1])],
        "map_xy_m": _interpolate_map_centerline(aisle, t),
        "heading_rad": heading,
        "boundary_source": _boundary_source(
            point,
            distance_hard,
            distance_unknown,
            resolution,
        ),
    }


def estimate_aisle_handoff_boundary(
    base_map,
    aisle,
    resolution,
    radius_m=0.20,
):
    """Estimate the clearance-safe row core and entry/exit handoff poses.

    The selected row core is the connected component of ``safe & aisle`` that
    contains the geometric aisle midpoint. If the midpoint is not safe, the
    component with the largest longitudinal span is used as an explicit
    fallback. The component's longitudinal extrema define clearance-conditioned
    handoff boundaries; no map cell is edited.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be a 2D array")
    if float(resolution) <= 0.0:
        raise ValueError("resolution must be > 0")
    if float(radius_m) < 0.0:
        raise ValueError("radius_m must be >= 0")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")

    centerline = np.asarray(aisle.get("centerline_xy"), dtype=float)
    if centerline.shape != (2, 2):
        raise ValueError("aisle centerline_xy must be 2x2")
    full_start = centerline[0]
    full_end = centerline[1]
    vector = full_end - full_start
    norm2 = float(vector @ vector)
    length_cells = float(np.sqrt(norm2))
    if length_cells <= 1e-12:
        raise ValueError("aisle centerline length must be non-zero")
    length_m = float(aisle.get("length_m", length_cells * float(resolution)))

    free = base == FREE_VALUE
    hard = base == OCCUPIED_VALUE
    unknown = base == UNKNOWN_VALUE
    distance_to_nonfree = ndimage.distance_transform_edt(free) * float(resolution)
    distance_hard = _distance_to_mask(hard, resolution)
    distance_unknown = _distance_to_mask(unknown, resolution)
    safe = free & (distance_to_nonfree + 1e-12 >= float(radius_m))

    aisle_mask = rasterize_aisles([aisle], base.shape)
    labels, component_count = ndimage.label(safe & aisle_mask)
    if component_count == 0:
        return {
            "aisle_id": int(aisle.get("aisle_id", 0)),
            "label": str(aisle.get("label", "")),
            "radius_m": float(radius_m),
            "status": "no_safe_component",
            "component_selection": None,
            "component_id": 0,
            "safe_component_count": 0,
            "entry_handoff": None,
            "exit_handoff": None,
            "row_core_start_s_over_l": None,
            "row_core_end_s_over_l": None,
            "row_core_length_m": 0.0,
            "entry_transition_length_m": None,
            "exit_transition_length_m": None,
        }

    midpoint = 0.5 * (full_start + full_end)
    midpoint_label = _point_label(labels, midpoint)
    component_selection = "midpoint"
    selected_label = midpoint_label

    yy, xx = np.nonzero(aisle_mask)
    points = np.column_stack((xx.astype(float), yy.astype(float)))
    t_all = ((points - full_start) @ vector) / norm2

    if selected_label == 0:
        component_selection = "largest_longitudinal_span"
        best_key = None
        selected_label = 0
        for label_id in range(1, int(component_count) + 1):
            local = labels[yy, xx] == label_id
            if not np.any(local):
                continue
            local_t = t_all[local]
            span = float(np.max(local_t) - np.min(local_t))
            cell_count = int(np.count_nonzero(local))
            key = (span, cell_count)
            if best_key is None or key > best_key:
                best_key = key
                selected_label = int(label_id)

    if selected_label <= 0:
        raise RuntimeError("safe components exist but none could be selected")

    selected = labels[yy, xx] == int(selected_label)
    selected_t = t_all[selected]
    if selected_t.size == 0:
        raise RuntimeError("selected safe component has no aisle cells")

    start_t = float(np.clip(np.min(selected_t), 0.0, 1.0))
    end_t = float(np.clip(np.max(selected_t), 0.0, 1.0))
    if end_t < start_t:
        start_t, end_t = end_t, start_t

    entry = _handoff_record(
        aisle,
        start_t,
        full_start,
        vector,
        length_m,
        distance_hard,
        distance_unknown,
        resolution,
    )
    exit_ = _handoff_record(
        aisle,
        end_t,
        full_start,
        vector,
        length_m,
        distance_hard,
        distance_unknown,
        resolution,
    )

    return {
        "aisle_id": int(aisle.get("aisle_id", 0)),
        "label": str(aisle.get("label", "")),
        "radius_m": float(radius_m),
        "status": "ok",
        "component_selection": component_selection,
        "component_id": int(selected_label),
        "safe_component_count": int(component_count),
        "component_cell_count": int(np.count_nonzero(selected)),
        "row_core_start_s_over_l": start_t,
        "row_core_end_s_over_l": end_t,
        "row_core_length_m": max(0.0, (end_t - start_t) * length_m),
        "entry_transition_length_m": max(0.0, start_t * length_m),
        "exit_transition_length_m": max(0.0, (1.0 - end_t) * length_m),
        "entry_handoff": entry,
        "exit_handoff": exit_,
    }
