"""Localize clearance-connectivity failures inside recovered row aisles."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    rasterize_aisles,
)


def select_unexpected_failure_targets(diagnostics):
    """Return aisle label -> first geometrically unexpected failed radius."""
    targets = {}
    for item in diagnostics.get("aisles", []):
        radius = item.get("first_unexpected_failed_radius_m")
        if radius is not None:
            targets[str(item["label"])] = float(radius)
    return targets


def _probe_points(rectangle, fraction):
    polygon = np.asarray(rectangle["polygon_xy"], dtype=float)
    if polygon.shape != (4, 2):
        raise ValueError("aisle polygon must be 4x2")
    start = 0.5 * (polygon[0] + polygon[3])
    end = 0.5 * (polygon[1] + polygon[2])
    return start, end, start + fraction * (end - start), end - fraction * (end - start)


def _point_label(labels, xy):
    x, y = np.rint(xy).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _distance_to_mask(mask, resolution):
    mask = np.asarray(mask, dtype=bool)
    if np.any(mask):
        return ndimage.distance_transform_edt(~mask) * float(resolution)
    return np.full(mask.shape, np.inf, dtype=float)


def _region_from_t(t):
    if t < 0.25:
        return "entry"
    if t > 0.75:
        return "exit"
    return "interior"


def _dominant_source(sources):
    sources = [source for source in sources if source in {"hard", "unknown", "mixed"}]
    if not sources:
        return "undetermined"
    kinds = set(sources)
    if kinds == {"hard"}:
        return "hard"
    if kinds == {"unknown"}:
        return "unknown"
    return "mixed"


def _contiguous_segments(indices):
    values = sorted(int(value) for value in indices)
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def localize_clearance_blocker(
    base_map,
    aisle,
    resolution,
    radius_m,
    probe_fraction=0.10,
):
    """Reproduce clearance validation and localize the blocking longitudinal zone.

    Longitudinal slices are one grid cell wide in the aisle direction. A slice
    is clearance-blocked when no free cell in that slice reaches ``radius_m``
    distance from both occupied and unknown map cells.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be a 2D array")
    if float(resolution) <= 0.0 or float(radius_m) < 0.0:
        raise ValueError("resolution must be > 0 and radius_m must be >= 0")
    if not 0.0 <= float(probe_fraction) < 0.5:
        raise ValueError("probe_fraction must be in [0, 0.5)")

    free = base == FREE_VALUE
    hard = base == OCCUPIED_VALUE
    unknown = base == UNKNOWN_VALUE
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")

    distance_hard = _distance_to_mask(hard, resolution)
    distance_unknown = _distance_to_mask(unknown, resolution)
    clearance = np.minimum(distance_hard, distance_unknown)
    safe = free & (clearance + 1e-12 >= float(radius_m))

    aisle_mask = rasterize_aisles([aisle], base.shape)
    labels, _ = ndimage.label(safe & aisle_mask)
    full_start, full_end, start_probe, end_probe = _probe_points(
        aisle, float(probe_fraction)
    )
    start_label = _point_label(labels, start_probe)
    end_label = _point_label(labels, end_probe)
    validation_pass = start_label > 0 and start_label == end_label

    vector = full_end - full_start
    length_cells = float(np.linalg.norm(vector))
    if length_cells <= 1e-12:
        raise ValueError("aisle centerline length must be non-zero")
    length_m = float(aisle.get("length_m", length_cells * float(resolution)))
    norm2 = float(vector @ vector)

    yy, xx = np.nonzero(aisle_mask)
    points = np.column_stack((xx.astype(float), yy.astype(float)))
    t = ((points - full_start) @ vector) / norm2
    keep = (t >= float(probe_fraction)) & (t <= 1.0 - float(probe_fraction))
    points = points[keep]
    yy = yy[keep]
    xx = xx[keep]
    t = t[keep]
    if t.size == 0:
        raise ValueError("aisle contains no cells inside the probe interval")

    longitudinal_cell = np.floor((t - float(probe_fraction)) * length_cells).astype(int)
    records = []
    for bin_id in sorted(int(value) for value in np.unique(longitudinal_cell)):
        local = longitudinal_cell == bin_id
        local_y = yy[local]
        local_x = xx[local]
        local_free = free[local_y, local_x]
        local_clearance = clearance[local_y, local_x]

        if np.any(local_free):
            free_indices = np.flatnonzero(local_free)
            best_local = free_indices[int(np.argmax(local_clearance[local_free]))]
            by = int(local_y[best_local])
            bx = int(local_x[best_local])
            max_clearance = float(clearance[by, bx])
            dh = float(distance_hard[by, bx])
            du = float(distance_unknown[by, bx])
            tolerance = 0.25 * float(resolution)
            if abs(dh - du) <= tolerance:
                source = "mixed"
            elif dh < du:
                source = "hard"
            else:
                source = "unknown"
        else:
            max_clearance = 0.0
            hard_count = int(np.count_nonzero(hard[local_y, local_x]))
            unknown_count = int(np.count_nonzero(unknown[local_y, local_x]))
            if hard_count and unknown_count:
                source = "mixed"
            elif hard_count:
                source = "hard"
            elif unknown_count:
                source = "unknown"
            else:
                source = "undetermined"

        bin_t0 = float(probe_fraction) + bin_id / length_cells
        bin_t1 = float(probe_fraction) + (bin_id + 1) / length_cells
        records.append({
            "bin_id": bin_id,
            "start_s_over_l": max(float(probe_fraction), bin_t0),
            "end_s_over_l": min(1.0 - float(probe_fraction), bin_t1),
            "max_clearance_m": max_clearance,
            "clearance_blocked": bool(max_clearance + 1e-12 < float(radius_m)),
            "blocking_source": source,
        })

    blocked_ids = [item["bin_id"] for item in records if item["clearance_blocked"]]
    by_id = {item["bin_id"]: item for item in records}
    segments = []
    for group in _contiguous_segments(blocked_ids):
        first = by_id[group[0]]
        last = by_id[group[-1]]
        start_t = float(first["start_s_over_l"])
        end_t = float(last["end_s_over_l"])
        mid_t = 0.5 * (start_t + end_t)
        sources = [by_id[index]["blocking_source"] for index in group]
        segment_cells = (t >= start_t - 1e-12) & (t <= end_t + 1e-12)
        segment_y = yy[segment_cells]
        segment_x = xx[segment_cells]
        hard_cells = int(np.count_nonzero(hard[segment_y, segment_x]))
        unknown_cells = int(np.count_nonzero(unknown[segment_y, segment_x]))
        if hard_cells and unknown_cells:
            segment_source = "mixed"
        elif hard_cells:
            segment_source = "hard"
        elif unknown_cells:
            segment_source = "unknown"
        else:
            segment_source = _dominant_source(sources)
        segments.append({
            "start_s_over_l": start_t,
            "end_s_over_l": end_t,
            "start_s_m": start_t * length_m,
            "end_s_m": end_t * length_m,
            "length_m": max(0.0, (end_t - start_t) * length_m),
            "region": _region_from_t(mid_t),
            "dominant_blocking_source": segment_source,
            "hard_cell_count": hard_cells,
            "unknown_cell_count": unknown_cells,
            "min_cross_section_clearance_m": float(
                min(by_id[index]["max_clearance_m"] for index in group)
            ),
        })

    first_blocker = segments[0] if segments else None
    longest_blocker = max(segments, key=lambda item: item["length_m"], default=None)

    disconnect_mode = "connected" if validation_pass else "undetermined"
    failure_t = None
    if not validation_pass:
        if start_label == 0:
            disconnect_mode = "entry_probe_blocked"
            failure_t = float(probe_fraction)
        elif end_label == 0:
            disconnect_mode = "exit_probe_blocked"
            failure_t = 1.0 - float(probe_fraction)
        else:
            cell_t = ((points - full_start) @ vector) / norm2
            cell_labels = labels[yy, xx]
            start_t = cell_t[cell_labels == start_label]
            end_t = cell_t[cell_labels == end_label]
            if start_t.size and end_t.size:
                start_max = float(np.max(start_t))
                end_min = float(np.min(end_t))
                if start_max < end_min:
                    disconnect_mode = "longitudinal_gap"
                    failure_t = 0.5 * (start_max + end_min)
                else:
                    disconnect_mode = "lateral_disconnect"
                    failure_t = 0.5 * (end_min + start_max)

    if longest_blocker is not None:
        failure_region = str(longest_blocker["region"])
        dominant_source = str(longest_blocker["dominant_blocking_source"])
    elif failure_t is not None:
        failure_region = _region_from_t(failure_t)
        x = int(round(full_start[0] + failure_t * vector[0]))
        y = int(round(full_start[1] + failure_t * vector[1]))
        if 0 <= y < base.shape[0] and 0 <= x < base.shape[1]:
            dh = float(distance_hard[y, x])
            du = float(distance_unknown[y, x])
            tolerance = 0.25 * float(resolution)
            if abs(dh - du) <= tolerance:
                dominant_source = "mixed"
            elif dh < du:
                dominant_source = "hard"
            else:
                dominant_source = "unknown"
        else:
            dominant_source = "undetermined"
    else:
        failure_region = None
        dominant_source = "undetermined"

    return {
        "aisle_id": int(aisle.get("aisle_id", 0)),
        "label": str(aisle.get("label", "")),
        "radius_m": float(radius_m),
        "probe_fraction": float(probe_fraction),
        "validation_pass": bool(validation_pass),
        "start_probe_safe": bool(start_label > 0),
        "end_probe_safe": bool(end_label > 0),
        "start_component_id": int(start_label),
        "end_component_id": int(end_label),
        "disconnect_mode": disconnect_mode,
        "failure_region": failure_region,
        "dominant_blocking_source": dominant_source,
        "first_blocker": first_blocker,
        "longest_blocker": longest_blocker,
        "blocker_segments": segments,
    }
