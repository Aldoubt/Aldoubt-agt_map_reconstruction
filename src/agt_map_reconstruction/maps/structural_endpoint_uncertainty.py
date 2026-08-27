"""Uncertainty-preserving structural endpoint envelope for P1-D3.1 v3.

Unlike the earlier bilateral gate, this module keeps every ridge termination
that has direct structural evidence.  Aisle-level left/right disagreement is
reported as uncertainty metadata instead of deleting the underlying ridge
observations.
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


def _robust_line(v, u):
    v = np.asarray(v, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    slopes = []
    for i in range(v.size):
        for j in range(i + 1, v.size):
            dv = float(v[j] - v[i])
            if abs(dv) <= 1e-12:
                continue
            slopes.append(float((u[j] - u[i]) / dv))
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(u - slope * v))
    return slope, intercept


def _quantiles(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"p50": None, "p90": None, "p95": None, "max": None}
    return {
        "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _side_payload(ridge_terminations, side, axis, cross, resolution_m):
    points = []
    ridge_points = []
    unsupported = []
    for ridge in ridge_terminations:
        ridge_id = str(ridge.get("ridge_id", ""))
        if ridge.get("status") != "ok":
            unsupported.append(ridge_id)
            continue
        point = ridge.get(f"{side}_grid_xy")
        u = ridge.get(f"{side}_u_cells")
        if point is None or u is None:
            unsupported.append(ridge_id)
            continue
        xy = np.asarray(point, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(f"{ridge_id} {side}_grid_xy must contain two values")
        points.append(xy)
        ridge_points.append(
            {
                "ridge_id": ridge_id,
                "grid_xy": [float(xy[0]), float(xy[1])],
                "u_cells": float(u),
                "cross_v_cells": float(xy @ cross),
                "residual_m": None,
                "abs_residual_m": None,
            }
        )

    if not points:
        return {
            "supported_count": 0,
            "unsupported_count": len(unsupported),
            "supported_fraction": 0.0,
            "trend_status": "insufficient_support",
            "trend": None,
            "abs_residual_m": _quantiles([]),
            "cross_row_span_fraction": 0.0,
            "ridge_points": [],
            "unsupported_ridge_ids": unsupported,
        }

    stacked = np.stack(points, axis=0)
    u = stacked @ axis
    v = stacked @ cross
    slope, intercept = _robust_line(v, u)
    residual_m = (u - (slope * v + intercept)) * float(resolution_m)
    abs_residual = np.abs(residual_m)
    for record, residual in zip(ridge_points, residual_m):
        record["residual_m"] = float(residual)
        record["abs_residual_m"] = float(abs(residual))

    if v.size >= 2:
        supported_span = float(np.ptp(v))
    else:
        supported_span = 0.0

    all_cross = []
    for ridge in ridge_terminations:
        for candidate_side in ("entry", "exit"):
            point = ridge.get(f"{candidate_side}_grid_xy")
            if point is None:
                continue
            xy = np.asarray(point, dtype=np.float64)
            if xy.shape == (2,):
                all_cross.append(float(xy @ cross))
                break
    total_span = float(np.ptp(np.asarray(all_cross, dtype=np.float64))) if len(all_cross) >= 2 else 0.0
    span_fraction = 1.0 if total_span <= 1e-12 and v.size > 0 else (
        0.0 if total_span <= 1e-12 else min(1.0, supported_span / total_span)
    )

    supported = len(ridge_points)
    total = len(ridge_terminations)
    return {
        "supported_count": supported,
        "unsupported_count": len(unsupported),
        "supported_fraction": 0.0 if total == 0 else float(supported / total),
        "trend_status": "ok" if supported >= 2 else "single_support",
        "trend": {
            "method": "median_pairwise_slope_plus_median_intercept",
            "slope_du_dv": slope,
            "intercept_u": intercept,
            "center_trend_only": True,
        },
        "abs_residual_m": _quantiles(abs_residual),
        "cross_row_span_fraction": float(span_fraction),
        "ridge_points": ridge_points,
        "unsupported_ridge_ids": unsupported,
    }


def _aisle_uncertainty(records):
    output = []
    for record in records:
        item = {
            "label": str(record.get("label", "")),
            "left_ridge_id": record.get("left_ridge_id"),
            "right_ridge_id": record.get("right_ridge_id"),
        }
        for side in ("entry", "exit"):
            source = dict(record.get(side) or {})
            disagreement = source.get("side_disagreement_m")
            candidate = source.get("candidate_grid_xy")
            status = str(source.get("status", ""))
            if disagreement is not None:
                evidence_class = "bilateral_agree" if status == "ok_bilateral" else "bilateral_disagree"
            elif candidate is not None:
                evidence_class = "single_side"
            else:
                evidence_class = "no_structural_support"
            item[side] = {
                "source_status": status,
                "evidence_class": evidence_class,
                "candidate_grid_xy": candidate,
                "candidate_u_cells": source.get("candidate_u_cells"),
                "side_disagreement_m": None if disagreement is None else float(disagreement),
            }
        output.append(item)
    return output


def build_structural_endpoint_uncertainty_envelope(structural_bundle):
    """Build a structural endpoint trend plus explicit uncertainty metadata."""
    bundle = dict(structural_bundle)
    ridges = list(bundle.get("ridge_terminations") or [])
    axis = _unit(bundle.get("row_axis_direction"))
    cross = _unit(bundle.get("cross_row_direction"))
    if abs(float(axis @ cross)) > 1e-6:
        cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    resolution = float(bundle.get("resolution_m", 0.0))
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")

    supported = sum(1 for item in ridges if item.get("status") == "ok")
    result = {
        "schema_version": 1,
        "method": "ridge_termination_uncertainty_envelope",
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "resolution_m": resolution,
        "ridge_count": len(ridges),
        "supported_ridge_count": supported,
        "unsupported_ridge_count": len(ridges) - supported,
        "entry": _side_payload(ridges, "entry", axis, cross, resolution),
        "exit": _side_payload(ridges, "exit", axis, cross, resolution),
        "aisle_endpoint_uncertainty": _aisle_uncertainty(bundle.get("paired_endpoints") or []),
        "policy": {
            "ridge_outliers_deleted": False,
            "bilateral_agreement_required_for_envelope": False,
            "side_disagreement_preserved": True,
            "center_trend_is_semantic_boundary": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result
