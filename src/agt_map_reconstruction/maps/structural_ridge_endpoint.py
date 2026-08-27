"""Inter-aisle ridge profiles and boundary-anchored structural endpoints for D3.1 v2."""

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
    return {
        "label": str(row.get("label", "")),
        "u_min": float(np.min(pu)),
        "u_max": float(np.max(pu)),
        "v_min": float(np.min(pv)),
        "v_max": float(np.max(pv)),
        "center_v": float(np.mean(line @ cross)),
    }


def build_inter_aisle_ridge_profiles(
    base_map,
    rows,
    *,
    resolution_m,
    bin_size_m,
    row_axis=None,
):
    """Build HARD/UNKNOWN profiles inside bands between adjacent aisle polygons."""
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
        axis = _unit(np.mean(np.stack(directions), axis=0))
    else:
        axis = _unit(row_axis)
    cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    geom = sorted((_row_geometry(row, axis, cross) for row in rows), key=lambda x: x["center_v"])

    yy, xx = np.indices(base.shape)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    all_u = points @ axis
    all_v = points @ cross
    values = base.ravel()
    bin_size_cells = bin_size / resolution

    profiles = []
    for lower, upper in zip(geom[:-1], geom[1:]):
        v0, v1 = float(lower["v_max"]), float(upper["v_min"])
        u0 = max(float(lower["u_min"]), float(upper["u_min"]))
        u1 = min(float(lower["u_max"]), float(upper["u_max"]))
        if v1 <= v0 + 1e-9 or u1 <= u0 + 1e-9:
            continue

        region = (
            (all_v > v0 + 1e-12)
            & (all_v < v1 - 1e-12)
            & (all_u >= u0 - 1e-12)
            & (all_u <= u1 + 1e-12)
        )
        count = int(np.ceil((u1 - u0) / bin_size_cells))
        edges = u0 + np.arange(count + 1, dtype=np.float64) * bin_size_cells
        edges[-1] = u1
        centers = 0.5 * (edges[:-1] + edges[1:])
        center_v = 0.5 * (v0 + v1)
        center_xy = centers[:, None] * axis[None, :] + center_v * cross[None, :]

        hard_fraction, unknown_fraction, cell_count = [], [], []
        for index in range(count):
            lo, hi = float(edges[index]), float(edges[index + 1])
            in_bin = (all_u >= lo - 1e-12) & (
                (all_u <= hi + 1e-12) if index + 1 == count else (all_u < hi - 1e-12)
            )
            sample = values[region & in_bin]
            n = int(sample.size)
            cell_count.append(n)
            hard_fraction.append(0.0 if n == 0 else float(np.count_nonzero(sample == OCCUPIED_VALUE) / n))
            unknown_fraction.append(0.0 if n == 0 else float(np.count_nonzero(sample == UNKNOWN_VALUE) / n))

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
    i = 0
    while i < values.size:
        if values[i]:
            i += 1
            continue
        start = i
        while i < values.size and not values[i]:
            i += 1
        if (
            max_gap_bins > 0
            and start > 0
            and i < values.size
            and values[start - 1]
            and values[i]
            and i - start <= max_gap_bins
        ):
            values[start:i] = True
    return values


def _first_sustained_from_edge(mask, min_bins, side):
    values = np.asarray(mask, dtype=bool)
    n = int(values.size)
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
    """Find first sustained ridge support while scanning inward from each end."""
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

    common = {
        "schema_version": 2,
        "ridge_id": profile["ridge_id"],
        "left_aisle_label": profile["left_aisle_label"],
        "right_aisle_label": profile["right_aisle_label"],
        "resolution_m": resolution,
        "support_mask": closed.tolist(),
    }
    if entry_bin is None or exit_bin is None or entry_bin >= exit_bin:
        return {
            **common,
            "status": "insufficient_structural_support",
            "entry_u_cells": None,
            "exit_u_cells": None,
            "entry_grid_xy": None,
            "exit_grid_xy": None,
        }

    entry_u, exit_u = float(edges[entry_bin]), float(edges[exit_bin])
    axis = np.asarray(profile["row_axis_direction"], dtype=np.float64)
    cross = np.asarray(profile["cross_row_direction"], dtype=np.float64)
    v0, v1 = profile["ridge_cross_span_cells"]
    center_v = 0.5 * (float(v0) + float(v1))

    def point(u):
        xy = u * axis + center_v * cross
        return [float(xy[0]), float(xy[1])]

    return {
        **common,
        "status": "ok",
        "entry_u_cells": entry_u,
        "exit_u_cells": exit_u,
        "entry_grid_xy": point(entry_u),
        "exit_grid_xy": point(exit_u),
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
    ridges = {
        (item["left_aisle_label"], item["right_aisle_label"]): item
        for item in ridge_terminations
    }
    maximum = float(max_side_endpoint_disagreement_m)

    def make_side(left_ridge, right_ridge, side):
        available = []
        for ridge in (left_ridge, right_ridge):
            if ridge is None or ridge.get("status") != "ok":
                continue
            u = ridge.get(f"{side}_u_cells")
            xy = ridge.get(f"{side}_grid_xy")
            if u is not None and xy is not None:
                available.append((float(u), np.asarray(xy, dtype=np.float64), float(ridge["resolution_m"])))
        if not available:
            return {
                "status": "insufficient_structural_support",
                "structural_u_cells": None,
                "structural_grid_xy": None,
                "candidate_u_cells": None,
                "candidate_grid_xy": None,
                "side_disagreement_m": None,
            }
        if len(available) == 1:
            u, xy, _ = available[0]
            return {
                "status": "ambiguous_single_side",
                "structural_u_cells": None,
                "structural_grid_xy": None,
                "candidate_u_cells": u,
                "candidate_grid_xy": xy.tolist(),
                "side_disagreement_m": None,
            }
        (u0, p0, r0), (u1, p1, r1) = available
        if not np.isclose(r0, r1):
            raise ValueError("neighboring ridge resolutions do not match")
        disagreement = abs(u0 - u1) * r0
        candidate_u = 0.5 * (u0 + u1)
        candidate_xy = 0.5 * (p0 + p1)
        ok = disagreement <= maximum + 1e-12
        return {
            "status": "ok_bilateral" if ok else "ambiguous_single_side",
            "structural_u_cells": candidate_u if ok else None,
            "structural_grid_xy": candidate_xy.tolist() if ok else None,
            "candidate_u_cells": candidate_u,
            "candidate_grid_xy": candidate_xy.tolist(),
            "side_disagreement_m": disagreement,
        }

    results = []
    for index, item in enumerate(geom):
        label = item["label"]
        left = None if index == 0 else ridges.get((geom[index - 1]["label"], label))
        right = None if index + 1 == len(geom) else ridges.get((label, geom[index + 1]["label"]))
        results.append(
            {
                "label": label,
                "entry": make_side(left, right, "entry"),
                "exit": make_side(left, right, "exit"),
                "left_ridge_id": None if left is None else left["ridge_id"],
                "right_ridge_id": None if right is None else right["ridge_id"],
            }
        )
    return results
