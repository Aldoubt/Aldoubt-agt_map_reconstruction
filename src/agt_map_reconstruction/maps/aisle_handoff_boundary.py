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


def _map_point(metadata, point_xy):
    if metadata is None:
        return None
    x, y = np.asarray(point_xy, dtype=float)
    world = metadata.grid_to_world(float(x), float(y))
    return [float(world[0]), float(world[1])]


def _choose_boundary_cell(
    selected_points,
    selected_t,
    full_start,
    vector,
    direction,
    normal,
    distance_to_nonfree,
    resolution,
    side,
):
    """Choose a real safe cell near one longitudinal component boundary.

    The longitudinal extremum defines the boundary station. Within one grid-cell
    longitudinal tolerance, prefer the cell with the largest clearance and then
    the smallest absolute cross-track offset. This avoids projecting a component
    boundary back onto an unsafe geometric centreline.
    """
    if side not in {"entry", "exit"}:
        raise ValueError("side must be entry or exit")

    extremum = float(np.min(selected_t) if side == "entry" else np.max(selected_t))
    length_cells = float(np.linalg.norm(vector))
    longitudinal_tolerance = 1.0 / max(length_cells, 1.0) + 1e-12
    if side == "entry":
        candidates = selected_t <= extremum + longitudinal_tolerance
    else:
        candidates = selected_t >= extremum - longitudinal_tolerance

    candidate_points = selected_points[candidates]
    candidate_t = selected_t[candidates]
    if candidate_points.size == 0:
        raise RuntimeError("selected component boundary has no candidate cells")

    projected = full_start + candidate_t[:, None] * vector
    cross_track_cells = (candidate_points - projected) @ normal
    cross_track_m = cross_track_cells * float(resolution)

    cx = np.rint(candidate_points[:, 0]).astype(int)
    cy = np.rint(candidate_points[:, 1]).astype(int)
    clearance = distance_to_nonfree[cy, cx]

    # Lexicographic objective: maximize clearance, then stay closest to the
    # centreline, then use the most extreme longitudinal station.
    if side == "entry":
        longitudinal_tie = candidate_t
    else:
        longitudinal_tie = -candidate_t
    order = np.lexsort((longitudinal_tie, np.abs(cross_track_m), -clearance))
    index = int(order[0])
    return {
        "point": candidate_points[index],
        "t": float(candidate_t[index]),
        "cross_track_offset_m": float(cross_track_m[index]),
        "clearance_m": float(clearance[index]),
    }


def _handoff_record(
    aisle,
    boundary,
    length_m,
    distance_hard,
    distance_unknown,
    resolution,
    metadata=None,
):
    point = np.asarray(boundary["point"], dtype=float)
    t = float(boundary["t"])
    heading = float(aisle.get("heading_rad", 0.0))
    return {
        "s_over_l": t,
        "s_m": t * float(length_m),
        "grid_xy": [float(point[0]), float(point[1])],
        "map_xy_m": _map_point(metadata, point),
        "heading_rad": heading,
        "cross_track_offset_m": float(boundary["cross_track_offset_m"]),
        "clearance_m": float(boundary["clearance_m"]),
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
    metadata=None,
):
    """Estimate the clearance-safe row core and entry/exit handoff poses.

    The selected row core is the connected component of ``safe & aisle`` that
    contains the geometric aisle midpoint. If the midpoint is not safe, the
    component with the largest longitudinal span is used as an explicit
    fallback. Handoff poses are actual cells on the selected safe component,
    not projections onto the geometric centreline, so a laterally shifted safe
    exit can be represented when the centreline probe is blocked.

    No map cell is edited. ``metadata`` is optional; when supplied, exact
    map-frame handoff coordinates are emitted through ``GridMetadata``.
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
    direction = vector / length_cells
    normal = np.array([-direction[1], direction[0]], dtype=float)
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
    selected_points = points[selected]
    selected_t = t_all[selected]
    if selected_t.size == 0:
        raise RuntimeError("selected safe component has no aisle cells")

    entry_boundary = _choose_boundary_cell(
        selected_points,
        selected_t,
        full_start,
        vector,
        direction,
        normal,
        distance_to_nonfree,
        resolution,
        side="entry",
    )
    exit_boundary = _choose_boundary_cell(
        selected_points,
        selected_t,
        full_start,
        vector,
        direction,
        normal,
        distance_to_nonfree,
        resolution,
        side="exit",
    )
    start_t = float(np.clip(entry_boundary["t"], 0.0, 1.0))
    end_t = float(np.clip(exit_boundary["t"], 0.0, 1.0))
    if end_t < start_t:
        entry_boundary, exit_boundary = exit_boundary, entry_boundary
        start_t, end_t = end_t, start_t

    entry = _handoff_record(
        aisle,
        entry_boundary,
        length_m,
        distance_hard,
        distance_unknown,
        resolution,
        metadata=metadata,
    )
    exit_ = _handoff_record(
        aisle,
        exit_boundary,
        length_m,
        distance_hard,
        distance_unknown,
        resolution,
        metadata=metadata,
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
