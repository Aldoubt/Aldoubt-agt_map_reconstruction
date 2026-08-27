"""Localize trajectory-aware ray support inside frozen P1-D3 endpoint ROIs.

The diagnostics are descriptive only. They reuse the endpoint geometry already
stored in the frozen D3 envelope and never modify the canonical navigation map.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _side_roi(shape, baseline_envelope, side_name):
    row_axis = np.asarray(baseline_envelope["row_axis_direction"], dtype=np.float64)
    cross_axis = np.asarray(
        baseline_envelope.get("cross_row_direction", [-row_axis[1], row_axis[0]]),
        dtype=np.float64,
    )
    row_v_min, row_v_max = [float(v) for v in baseline_envelope["row_cross_span"]]
    fit = baseline_envelope["sides"][side_name]["endpoint_fit"]

    yy, xx = np.indices(shape)
    u = xx.astype(np.float64) * row_axis[0] + yy.astype(np.float64) * row_axis[1]
    v = xx.astype(np.float64) * cross_axis[0] + yy.astype(np.float64) * cross_axis[1]
    boundary_u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    cross_roi = (v >= row_v_min - 1e-12) & (v <= row_v_max + 1e-12)
    if side_name == "entry":
        outward = u < boundary_u - 1e-12
        outward_depth_cells = boundary_u - u
    elif side_name == "exit":
        outward = u > boundary_u + 1e-12
        outward_depth_cells = u - boundary_u
    else:
        raise ValueError("side_name must be entry or exit")
    return cross_roi & outward, v, outward_depth_cells


def _component_summary(mask):
    labels, count = ndimage.label(mask)
    if int(count) == 0:
        return {"component_count": 0, "largest_component_cell_count": 0}
    sizes = np.bincount(labels.reshape(-1))[1:]
    return {
        "component_count": int(count),
        "largest_component_cell_count": int(np.max(sizes)) if sizes.size else 0,
    }


def analyze_endpoint_support_threshold(
    base_map,
    support_count,
    baseline_envelope,
    *,
    min_support_rays,
    resolution,
):
    """Describe one support-count threshold inside the exact frozen D3 ROIs."""
    base = np.asarray(base_map, dtype=np.uint8)
    count = np.asarray(support_count)
    if base.ndim != 2 or count.shape != base.shape:
        raise ValueError("base_map/support_count shape mismatch")
    if int(min_support_rays) < 1:
        raise ValueError("min_support_rays must be >= 1")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")

    support = count >= int(min_support_rays)
    unknown = base == UNKNOWN_VALUE
    free = base == FREE_VALUE
    occupied = base == OCCUPIED_VALUE

    overlay_free = free | (support & unknown)
    radius_m = float(baseline_envelope["radius_m"])
    baseline_strict = free & (
        ndimage.distance_transform_edt(free) * float(resolution) + 1e-12 >= radius_m
    )
    candidate_strict = overlay_free & (
        ndimage.distance_transform_edt(overlay_free) * float(resolution) + 1e-12 >= radius_m
    )
    newly_strict = candidate_strict & ~baseline_strict

    result = {
        "min_support_rays": int(min_support_rays),
        "supported_cell_count": int(np.count_nonzero(support)),
        "supported_unknown_cell_count": int(np.count_nonzero(support & unknown)),
        "supported_existing_free_cell_count": int(np.count_nonzero(support & free)),
        "supported_occupied_cell_count_ignored": int(np.count_nonzero(support & occupied)),
        "new_strict_safe_cell_count": int(np.count_nonzero(newly_strict)),
        "sides": {},
        "automatic_acceptance": False,
        "semantic_promotion": False,
    }

    for side_name in ("entry", "exit"):
        roi, cross_v, outward_depth_cells = _side_roi(base.shape, baseline_envelope, side_name)
        roi_unknown = roi & unknown
        roi_supported_unknown = roi_unknown & support
        roi_new_strict = roi & newly_strict
        component = _component_summary(roi_supported_unknown)

        if np.any(roi_supported_unknown):
            local_v = cross_v[roi_supported_unknown]
            row_v_min, row_v_max = [float(v) for v in baseline_envelope["row_cross_span"]]
            row_span = max(1e-12, row_v_max - row_v_min)
            raw_cross_span_fraction = float(
                np.clip((float(np.max(local_v)) - float(np.min(local_v))) / row_span, 0.0, 1.0)
            )
            max_outward_depth_m = float(
                np.max(outward_depth_cells[roi_supported_unknown]) * float(resolution)
            )
        else:
            raw_cross_span_fraction = 0.0
            max_outward_depth_m = 0.0

        result["sides"][side_name] = {
            "roi_cell_count": int(np.count_nonzero(roi)),
            "roi_unknown_cell_count": int(np.count_nonzero(roi_unknown)),
            "supported_unknown_cell_count": int(np.count_nonzero(roi_supported_unknown)),
            "supported_unknown_fraction_of_roi_unknown": (
                float(np.count_nonzero(roi_supported_unknown) / np.count_nonzero(roi_unknown))
                if np.count_nonzero(roi_unknown) else 0.0
            ),
            "raw_supported_cross_row_span_fraction": raw_cross_span_fraction,
            "raw_supported_max_outward_depth_m": max_outward_depth_m,
            "new_strict_safe_cell_count": int(np.count_nonzero(roi_new_strict)),
            **component,
        }

    return result


def sweep_endpoint_support_thresholds(
    base_map,
    support_count,
    baseline_envelope,
    *,
    min_support_values,
    resolution,
):
    values = [int(v) for v in min_support_values]
    if not values or any(v < 1 for v in values):
        raise ValueError("min_support_values must contain positive integers")
    return {
        "schema_version": 1,
        "radius_m": float(baseline_envelope["radius_m"]),
        "thresholds": [
            analyze_endpoint_support_threshold(
                base_map,
                support_count,
                baseline_envelope,
                min_support_rays=value,
                resolution=resolution,
            )
            for value in values
        ],
        "automatic_threshold_selection": False,
        "semantic_promotion": False,
    }
