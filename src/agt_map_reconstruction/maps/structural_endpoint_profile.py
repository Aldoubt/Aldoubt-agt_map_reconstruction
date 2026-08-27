"""Build bilateral structural-support profiles beside recovered row aisles.

D3.1 treats structural row termination as a property of ridge / plant-side
structure, not of free-space reachability. This module therefore samples HARD
and UNKNOWN evidence in narrow strips immediately outside each aisle polygon.
UNKNOWN is reported explicitly and never counted as structural support.
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


def _fraction(values, target):
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values == target) / values.size)


def build_structural_support_profile(
    base_map,
    aisle,
    *,
    resolution_m,
    strip_width_m,
    bin_size_m,
    row_axis=None,
):
    """Sample HARD/UNKNOWN evidence in strips immediately beside one aisle.

    Parameters are explicit and intentionally not optimized automatically.
    Grid coordinates follow the repository convention where integer ``x,y``
    identify cell centres in the unflipped internal map array.
    """
    base = _validate_base_map(base_map)
    resolution = float(resolution_m)
    strip_width = float(strip_width_m)
    bin_size = float(bin_size_m)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")
    if strip_width <= 0.0:
        raise ValueError("strip_width_m must be > 0")
    if bin_size <= 0.0:
        raise ValueError("bin_size_m must be > 0")

    line = np.asarray(aisle.get("centerline_xy"), dtype=np.float64)
    polygon = np.asarray(aisle.get("polygon_xy"), dtype=np.float64)
    if line.shape != (2, 2):
        raise ValueError("aisle centerline_xy must be 2x2")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 4:
        raise ValueError("aisle polygon_xy must be Nx2 with at least 4 points")

    source_direction = _unit(line[1] - line[0])
    axis = source_direction if row_axis is None else _unit(row_axis)
    source_forward = bool(float((line[1] - line[0]) @ axis) >= 0.0)
    cross_axis = np.array([-axis[1], axis[0]], dtype=np.float64)

    polygon_u = polygon @ axis
    polygon_v = polygon @ cross_axis
    u_min = float(np.min(polygon_u))
    u_max = float(np.max(polygon_u))
    v_min = float(np.min(polygon_v))
    v_max = float(np.max(polygon_v))
    if u_max - u_min <= 1e-9:
        raise ValueError("aisle polygon has zero longitudinal extent")
    if v_max - v_min <= 1e-9:
        raise ValueError("aisle polygon has zero cross-row extent")

    strip_width_cells = strip_width / resolution
    bin_size_cells = bin_size / resolution
    bin_count = int(np.ceil((u_max - u_min) / bin_size_cells))
    if bin_count <= 0:
        raise ValueError("profile contains no longitudinal bins")

    yy, xx = np.indices(base.shape)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    u = points @ axis
    v = points @ cross_axis
    values = base.reshape(-1)

    longitudinal = (u >= u_min - 1e-12) & (u <= u_max + 1e-12)
    left_strip = (
        longitudinal
        & (v >= v_min - strip_width_cells - 1e-12)
        & (v < v_min - 1e-12)
    )
    right_strip = (
        longitudinal
        & (v > v_max + 1e-12)
        & (v <= v_max + strip_width_cells + 1e-12)
    )

    bin_edges = u_min + np.arange(bin_count + 1, dtype=np.float64) * bin_size_cells
    bin_edges[-1] = u_max
    center_u = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    center_v = 0.5 * (v_min + v_max)
    center_xy = center_u[:, None] * axis[None, :] + center_v * cross_axis[None, :]

    left_hard = []
    right_hard = []
    left_unknown = []
    right_unknown = []
    left_cell_count = []
    right_cell_count = []

    for index in range(bin_count):
        lo = float(bin_edges[index])
        hi = float(bin_edges[index + 1])
        if index + 1 == bin_count:
            in_bin = (u >= lo - 1e-12) & (u <= hi + 1e-12)
        else:
            in_bin = (u >= lo - 1e-12) & (u < hi - 1e-12)

        left_values = values[left_strip & in_bin]
        right_values = values[right_strip & in_bin]
        left_cell_count.append(int(left_values.size))
        right_cell_count.append(int(right_values.size))
        left_hard.append(_fraction(left_values, OCCUPIED_VALUE))
        right_hard.append(_fraction(right_values, OCCUPIED_VALUE))
        left_unknown.append(_fraction(left_values, UNKNOWN_VALUE))
        right_unknown.append(_fraction(right_values, UNKNOWN_VALUE))

    return {
        "schema_version": 1,
        "label": str(aisle.get("label", "")),
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross_axis.tolist(),
        "source_centerline_forward": source_forward,
        "longitudinal_span_cells": [u_min, u_max],
        "cross_row_span_cells": [v_min, v_max],
        "resolution_m": resolution,
        "strip_width_m": strip_width,
        "strip_width_cells": float(strip_width_cells),
        "bin_size_m": bin_size,
        "bin_size_cells": float(bin_size_cells),
        "bin_edges_u_cells": bin_edges.tolist(),
        "bin_center_u_cells": center_u.tolist(),
        "bin_center_grid_xy": center_xy.tolist(),
        "left_strip_cell_count": left_cell_count,
        "right_strip_cell_count": right_cell_count,
        "left_hard_support_fraction": left_hard,
        "right_hard_support_fraction": right_hard,
        "left_unknown_fraction": left_unknown,
        "right_unknown_fraction": right_unknown,
        "policy": {
            "hard_value": int(OCCUPIED_VALUE),
            "unknown_value": int(UNKNOWN_VALUE),
            "free_value": int(FREE_VALUE),
            "unknown_counted_as_structural": False,
            "automatic_parameter_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
