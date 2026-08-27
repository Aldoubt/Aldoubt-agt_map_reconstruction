"""Inter-aisle ridge profiles and boundary-anchored structural endpoints for D3.1 v2.

The v1 implementation sampled generic HARD cells beside each aisle and selected the
longest persistent run. In greenhouse maps this can latch onto walls, facilities or
an arbitrary internal occupied segment. V2 instead constructs explicit structural
bands between adjacent recovered aisles and finds the first sustained ridge support
when scanning inward from each longitudinal boundary.
"""

from __future__ import annotations

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def _validate_base_map(base_map):
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    return base


def _row_geometry(row, axis, cross):
    polygon = np.asarray(row.get("polygon_xy"), dtype=np.float64)
    line = np.asarray(row.get("centerline_xy"), dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 4:
        raise ValueError("row polygon_xy must be Nx2")
    if line.shape != (2, 2):
        raise ValueError("row centerline_xy must be 2x2")
    pu = polygon @ axis
    pv = polygon @ cross
    center_v = float(np.mean(line @ cross))
    return {
        "row": row,
        "label": str(row.get("label", "")),
        "u_min": float(np.min(pu)),
        "u_max": float(np.max(pu)),
        "v_min": float(np.min(pv)),
        "v_max": float(np.max(pv)),
        "center_v": center_v,
    }


def build_inter_aisle_ridge_profiles(
    base_map,
    rows,
    *,
    resolution_m,
    bin_size_m,
    row_axis=None,
):
    """Build HARD/UNKNOWN profiles in bands between adjacent row aisles."""
    base = _validate_base_map(base_map)
    resolution = float(resolution_m)
    bin_size = float(bin_size_m)
    if resolution <= 0.0 or bin_size <= 0.0:
        raise ValueError("resolution_m and bin_size_m must be > 0")

    rows = [item for item in rows if item.get("region_class", "row_aisle") == "row_aisle"]
    if len(rows) < 2:
        return []

    if row_axis is None:
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
        axis = _unit(np.mean(np.stack(directions, axis=0), axis=0))
    else:
        axis = _unit(row_axis)
    cross = np.array([-axis[1], axis[0]], dtype=np.float64)

    geom = sorted((_row_geometry(row, axis, cross) for row in rows), key=lambda x: x["center_v"])
    yy, xx = np.indices(base.shape)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    all_u = points @ axis
    all_v = points @ cross
    values = base.reshape(-1)
    bin_size_cells = bin_size / resolution

    profiles = []
    for index in range(len(geom) - 1):
        lower = geom[index]
        upper = geom[index + 1]
        v0 = float(lower["v_max"])
        v1 = float(upper["v_min"])
        u0 = max(float(lower["u_min"]), float(upper["u_min"]))
        u1 = min(float(lower["u_max"]), float(upper["u_max"]))
        if v1 <= v0 + 1e-9 or u1 <= u0 + 1e-9:
            continue

        ridge_mask = (
            (all_v > v0 + 1e-12)
            & (all_v < v1 - 1e-12)
            & (all_u >= u0 - 1e-12)
            & (all_u <= u1 + 1e-12)
        )
        bin_count = int(np.ceil((u1 - u0) / bin_size_cells))
        if bin_count <= 0:
            continue
        edges = u0 + np.arange(bin_count + 1, dtype=np.float64) * bin_size_cells
        edges[-1] = u1
        centers = 0.5 * (edges[:-1] + edges[1:])
        center_v = 0.5 * (v0 + v1)
        center_xy = centers[:, None] * axis[None, :] + center_v * cross[None, :]

        hard_fraction = []
        unknown_fraction = []
        cell_count = []
        for bin_index in range(bin_count):
            lo = float(edges[bin_index])
            hi = float(edges[bin_index + 1])
            if bin_index + 1 == bin_count:
                in_bin = (all_u >= lo - 1e-12) & (all_u <= hi + 1e-12)
            else:
                in_bin = (all_u >= lo - 1e-12) & (all_u < hi - 1e-12)
            sample = values[ridge_mask & in_bin]
            count = int(sample.size)
            cell_count.append(count)
            if count == 0:
                hard_fraction.append(0.0)
                unknown_fraction.append(0.0)
            else:
                hard_fraction.append(float(np.count_nonzero(sample == OCCUPIED_VALUE) / count))
                unknown_fraction.append(float(np.count_nonzero(sample == UNKNOWN_VALUE) / count))

        profiles.append(
            {
                "schema_version": 2,
                "ridge_id": f"R_{lower['label']}_{upper['label']}",
                "left_aisle_label": lower["label"],
                "right_aisle_label": upper["label"],
                "row_axis_direction": axis.tolist(),
                "cross_row_direction": cross.tolist(),
                "resolution_m": resolution,
                "bin_size_m": bin_size,
                "bin_edges_u_cells": edges.tolist(),
                "bin_center_u_cells": centers.tolist(),
                "bin_center_grid_xy": center_xy.tolist(),
                "ridge_cross_span_cells": [v0, v1],
                "ridge_cell_count": cell_count,
                "hard_support_fraction": hard_fraction,
                "unknown_fraction": unknown_fraction,
                "policy": {
                    "support_region": "between adjacent recovered aisle polygons",
                    "generic_outer_wall_used_as_ridge": False,
                    "unknown_counted_as_structural": False,
                    "navigation_map_modified": False,
                    "semantic_promotion": False,
                },
            }
        )
    return profiles


def _close_short_internal_gaps(mask, max_gap_bins):
    values = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < values.size:
        if values[index]:
            index += 1
            continue
        start = index
        while index < values.size and not values[index]:
            index += 1
        end = index
        if (
            max_gap_bins > 0
            and start > 0
            and end < values.size
            and values[start - 1]
            and values[end]
            and end - start <= max_gap_bins
        ):
            values[start:end] = True
    return values


def _first_sustained_from_edge(mask, min_bins, side):
    values = np.asarray(mask, dtype=bool)
    n = int(values.size)
    if n < min_bins:
        return None
    if side == "entry":
        for start in range(0, n - min_bins + 1):
            if bool(np.all(values[start : start + min_bins])):
                return int(start)
        return None
    if side == "exit":
        for end in range(n, min_bins - 1, -1):
            if bool(np.all(values[end - min_bins : end])):
                return int(end)
        return None
    raise ValueError("side must be entry or exit")


def detect_ridge_terminations(
    profile,
    *,
    min_support_fraction,
    min_persistence_m,
    max_internal_gap_m,
):
    """Detect ridge ends by scanning inward from both longitudinal boundaries."""
    support = np.asarray(profile["hard_support_fraction"], dtype=np.float64)
    edges = np.asarray(profile["bin_edges_u_cells"], dtype=np.float64)
    if support.ndim != 1 or edges.shape != (support.size + 1,):
        raise ValueError("ridge profile arrays have inconsistent shapes")
    threshold = float(min_support_fraction)
    persistence = float(min_persistence_m)
    gap = float(max_internal_gap_m)
    bin_size = float(profile["bin_size_m"])
    resolution = float(profile["resolution_m"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("min_support_fraction must be in (0,1]")
    if persistence <= 0.0 or gap < 0.0:
        raise ValueError("persistence must be >0 and gap >=0")

    min_bins = max(1, int(np.ceil(persistence / bin_size - 1e-12)))
    max_gap_bins = int(np.floor(gap / bin_size + 1e-12))
    raw = support + 1e-12 >= threshold
    closed = _close_short_internal_gaps(raw, max_gap_bins)
    entry_bin = _first_sustained_from_edge(closed, min_bins, "entry")
    exit_bin = _first_sustained_from_edge(closed, min_bins, "exit")
    if entry_bin is None or exit_bin is None or entry_bin >= exit_bin:
        return {
            "schema_version": 2,
            "ridge_id": profile["ridge_id"],
            "left_aisle_label": profile["left_aisle_label"],
            "right_aisle_label": profile["right_aisle_label"],
            "status": "insufficient_structural_support",
            "entry_u_cells": None,
            "exit_u_cells": None,
            "entry_grid_xy": None,
            "exit_grid_xy": None,
            "support_mask": closed.tolist(),
        }

    entry_u = float(edges[entry_bin])
    exit_u = float(edges[exit_bin])
    axis = np.asarray(profile["row_axis_direction"], dtype=np.float64)
    cross = np.asarray(profile["cross_row_direction"], dtype=np.float64)
    v0, v1 = profile["ridge_cross_span_cells"]
    center_v = 0.5 * (float(v0) + float(v1))

    def point(u):
        xy = float(u) * axis + center_v * cross
        return [float(xy[0]), float(xy[1])]

    return {
        "schema_version": 2,
        "ridge_id": profile["ridge_id"],
        "left_aisle_label": profile["left_aisle_label"],
        "right_aisle_label": profile["right_aisle_label"],
        "status": "ok",
        "entry_u_cells": entry_u,
        "exit_u_cells": exit_u,
        "entry_grid_xy": point(entry_u),
        "exit_grid_xy": point(exit_u),
        "support_mask": closed.tolist(),
        "parameters": {
            "min_support_fraction": threshold,
            "min_persistence_m": persistence,
            "max_internal_gap_m": gap,
            "min_persistence_bins": min_bins,
            "max_internal_gap_bins": max_gap_bins,
        },
        "policy": {
            "longest_internal_run_used": False,
            "boundary_anchored_detection": True,
            "unknown_counted_as_structural": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }


def pair_aisle_structural_endpoints(
    rows,
    ridge_terminations,
    *,
    row_axis,
    max_side_endpoint_disagreement_m,
):
    """Pair the two neighboring ridge terminations around each aisle."""
    axis = _unit(row_axis)
    cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    geom = sorted((_row_geometry(row, axis, cross) for row in rows), key=lambda x: x["center_v"])
    resolution = None
    ridges = {}
    for ridge in ridge_terminations:
        ridges[(ridge["left_aisle_label"], ridge["right_aisle_label"])] = ridge
    maximum = float(max_side_endpoint_disagreement_m)

    def make_side(left_ridge, right_ridge, side):
        values = []
        points = []
        for ridge in (left_ridge, right_ridge):
            if ridge is None or ridge.get("status") != "ok":
                continue
            value = ridge.get(f"{side}_u_cells")
            point = ridge.get(f"{side}_grid_xy")
            if value is not None and point is not None:
                values.append(float(value))
                points.append(np.asarray(point, dtype=np.float64))
        if not values:
            return {
                "status": "insufficient_structural_support",
                "structural_u_cells": None,
                "structural_grid_xy": None,
                "candidate_u_cells": None,
                "candidate_grid_xy": None,
                "side_disagreement_m": None,
            }
        if len(values) == 1:
            return {
                "status": "ambiguous_single_side",
                "structural_u_cells": None,
                "structural_grid_xy": None,
                "candidate_u_cells": values[0],
                "candidate_grid_xy": points[0].tolist(),
                "side_disagreement_m": None,
            }
        local_resolution = float(left_ridge.get("resolution_m", 1.0)) if left_ridge else 1.0
        # detect_ridge_terminations does not need to persist resolution; infer from
        # grid-cell disagreement using the source row-axis geometry only if absent.
        if "resolution_m" in left_ridge:
            local_resolution = float(left_ridge["resolution_m"])
        elif "resolution_m" in right_ridge:
            local_resolution = float(right_ridge["resolution_m"])
        else:
            local_resolution = 1.0
        disagreement = abs(values[0] - values[1]) * local_resolution
        candidate_u = 0.5 * (values[0] + values[1])
        candidate_xy = 0.5 * (points[0] + points[1])
        if disagreement > maximum + 1e-12:
            return {
                "status": "ambiguous_single_side",
                "structural_u_cells": None,
                "structural_grid_xy": None,
                "candidate_u_cells": candidate_u,
                "candidate_grid_xy": candidate_xy.tolist(),
                "side_disagreement_m": disagreement,
            }
        return {
            "status": "ok_bilateral",
            "structural_u_cells": candidate_u,
            "structural_grid_xy": candidate_xy.tolist(),
            "candidate_u_cells": candidate_u,
            "candidate_grid_xy": candidate_xy.tolist(),
            "side_disagreement_m": disagreement,
        }

    results = []
    for index, item in enumerate(geom):
        label = item["label"]
        left_ridge = None
        right_ridge = None
        if index > 0:
            left_label = geom[index - 1]["label"]
            left_ridge = ridges.get((left_label, label))
        if index + 1 < len(geom):
            right_label = geom[index + 1]["label"]
            right_ridge = ridges.get((label, right_label))
        results.append(
            {
                "label": label,
                "entry": make_side(left_ridge, right_ridge, "entry"),
                "exit": make_side(left_ridge, right_ridge, "exit"),
                "left_ridge_id": None if left_ridge is None else left_ridge["ridge_id"],
                "right_ridge_id": None if right_ridge is None else right_ridge["ridge_id"],
            }
        )
    return results
