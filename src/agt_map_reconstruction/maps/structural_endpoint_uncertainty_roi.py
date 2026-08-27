"""Uncertainty-aware ROI derived from fused P1-D3.1 structural endpoints.

The fused center trend is only a summary. Conservative headland ROIs begin
outside the trend +/- residual uncertainty band and explicitly exclude any
cross-row strip whose ridge endpoint remains structurally unresolved.

This module never changes the navigation map and never promotes geometry-only
lattice hypotheses or structural evidence to semantic free space.
"""

from __future__ import annotations

import numpy as np


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def _axes_and_resolution(fused_bundle, uncertainty_payload):
    fused_axis = _unit(fused_bundle.get("row_axis_direction"))
    fused_cross = _unit(fused_bundle.get("cross_row_direction"))
    unc_axis = _unit(uncertainty_payload.get("row_axis_direction"))
    unc_cross = _unit(uncertainty_payload.get("cross_row_direction"))
    if abs(float(fused_axis @ fused_cross)) > 1e-6:
        raise ValueError("fused row/cross axes must be orthogonal")
    if abs(float(unc_axis @ unc_cross)) > 1e-6:
        raise ValueError("uncertainty row/cross axes must be orthogonal")
    if not np.allclose(fused_axis, unc_axis, atol=1e-9, rtol=0.0):
        raise ValueError("fused and uncertainty row axes differ")
    if not np.allclose(fused_cross, unc_cross, atol=1e-9, rtol=0.0):
        raise ValueError("fused and uncertainty cross axes differ")
    fused_res = float(fused_bundle.get("resolution_m", 0.0))
    unc_res = float(uncertainty_payload.get("resolution_m", 0.0))
    if fused_res <= 0.0 or unc_res <= 0.0 or not np.isclose(fused_res, unc_res):
        raise ValueError("fused and uncertainty resolutions must match and be > 0")
    return fused_axis, fused_cross, fused_res


def _cross_domain(fused_bundle, cross_axis):
    values = []
    for row in fused_bundle.get("lattice_rows") or []:
        polygon = np.asarray(row.get("polygon_xy"), dtype=np.float64)
        if polygon.ndim == 2 and polygon.shape[1] == 2 and polygon.shape[0] >= 3:
            values.extend((polygon @ cross_axis).tolist())
    if not values:
        for profile in fused_bundle.get("ridge_profiles") or []:
            span = np.asarray(profile.get("ridge_cross_span_cells"), dtype=np.float64)
            if span.shape == (2,):
                values.extend(span.tolist())
    if len(values) < 2:
        raise ValueError("cannot resolve structural cross-row domain")
    return float(np.min(values)), float(np.max(values))


def _unresolved_cross_intervals(fused_bundle):
    profiles = {
        str(item.get("ridge_id", "")): item
        for item in fused_bundle.get("ridge_profiles") or []
    }
    intervals = []
    ids = []
    for termination in fused_bundle.get("ridge_terminations") or []:
        if termination.get("status") == "ok":
            continue
        ridge_id = str(termination.get("ridge_id", ""))
        profile = profiles.get(ridge_id)
        if profile is None:
            raise ValueError(f"missing ridge profile for unresolved ridge {ridge_id}")
        span = np.asarray(profile.get("ridge_cross_span_cells"), dtype=np.float64)
        if span.shape != (2,):
            raise ValueError(f"unresolved ridge {ridge_id} has invalid cross span")
        lo, hi = float(np.min(span)), float(np.max(span))
        intervals.append((lo, hi))
        ids.append(ridge_id)
    return ids, intervals


def _side_geometry(uncertainty_payload, side, resolution_m, quantile):
    payload = dict(uncertainty_payload.get(side) or {})
    if payload.get("trend_status") != "ok" or payload.get("trend") is None:
        raise ValueError(f"fused uncertainty {side} trend is not available")
    trend = payload["trend"]
    q = (payload.get("abs_residual_m") or {}).get(quantile)
    if q is None or not np.isfinite(float(q)) or float(q) < 0.0:
        raise ValueError(f"fused uncertainty {side} {quantile} residual is unavailable")
    return {
        "slope_du_dv": float(trend["slope_du_dv"]),
        "intercept_u": float(trend["intercept_u"]),
        "uncertainty_half_width_m": float(q),
        "uncertainty_half_width_cells": float(q) / float(resolution_m),
    }


def build_structural_endpoint_uncertainty_roi(
    fused_bundle,
    uncertainty_payload,
    *,
    grid_shape_yx,
    uncertainty_quantile="p95",
):
    """Build disjoint conservative, boundary-uncertain, and unresolved masks.

    Conservative and boundary-uncertainty cells are restricted to structurally
    resolved cross-row strips. Unresolved ridge strips are emitted only through
    ``structurally_unresolved_cross`` so downstream evidence counts do not
    silently double-count them.
    """
    shape = tuple(int(v) for v in grid_shape_yx)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("grid_shape_yx must be positive (height, width)")
    quantile = str(uncertainty_quantile)
    if quantile not in {"p50", "p90", "p95", "max"}:
        raise ValueError("uncertainty_quantile must be p50, p90, p95, or max")

    fused = dict(fused_bundle)
    uncertainty = dict(uncertainty_payload)
    axis, cross, resolution = _axes_and_resolution(fused, uncertainty)
    v_min, v_max = _cross_domain(fused, cross)
    unresolved_ids, unresolved_intervals = _unresolved_cross_intervals(fused)

    yy, xx = np.indices(shape)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    u = points @ axis
    v = points @ cross
    cross_domain = (v >= v_min - 1e-12) & (v <= v_max + 1e-12)

    unresolved_cross = np.zeros(v.shape, dtype=bool)
    for lo, hi in unresolved_intervals:
        unresolved_cross |= (v >= lo - 1e-12) & (v <= hi + 1e-12)
    unresolved_cross &= cross_domain
    resolved_cross = cross_domain & ~unresolved_cross

    masks = {"structurally_unresolved_cross": unresolved_cross.reshape(shape)}
    side_summaries = {}
    for side in ("entry", "exit"):
        geometry = _side_geometry(uncertainty, side, resolution, quantile)
        center = geometry["slope_du_dv"] * v + geometry["intercept_u"]
        half = geometry["uncertainty_half_width_cells"]
        uncertainty_band = resolved_cross & (np.abs(u - center) <= half + 1e-12)
        if side == "entry":
            conservative = resolved_cross & (u < center - half - 1e-12)
        else:
            conservative = resolved_cross & (u > center + half + 1e-12)
        masks[f"{side}_boundary_uncertainty"] = uncertainty_band.reshape(shape)
        masks[f"{side}_conservative_outward"] = conservative.reshape(shape)
        side_summaries[side] = {
            **geometry,
            "conservative_outward_cell_count": int(np.count_nonzero(conservative)),
            "boundary_uncertainty_cell_count": int(np.count_nonzero(uncertainty_band)),
        }

    result = {
        "schema_version": 1,
        "method": "fused_structural_endpoint_uncertainty_roi",
        "grid_shape_yx": list(shape),
        "resolution_m": resolution,
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "row_cross_span_cells": [v_min, v_max],
        "uncertainty_quantile": quantile,
        "unresolved_ridge_ids": unresolved_ids,
        "unresolved_ridge_count": len(unresolved_ids),
        "structurally_unresolved_cross_cell_count": int(np.count_nonzero(unresolved_cross)),
        "entry": side_summaries["entry"],
        "exit": side_summaries["exit"],
        "policy": {
            "uncertainty_width_source": "fused_ridge_abs_residual_quantile",
            "center_trend_promoted_to_semantic_boundary": False,
            "unresolved_cross_strip_excluded": True,
            "roi_partitions_structurally_disjoint": True,
            "geometry_only_lattice_supplies_structural_evidence": False,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result, masks
