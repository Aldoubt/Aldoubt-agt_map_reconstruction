"""Robust common-boundary fitting for D3.1 structural row terminations."""

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


def _robust_initial_line(v, u):
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


def _fit_side(
    records,
    side,
    *,
    row_axis,
    cross_axis,
    resolution_m,
    residual_floor_m,
    mad_scale,
    min_inlier_count,
    max_fit_rmse_m,
):
    rows = []
    candidate_points = []
    candidate_labels = []
    for record in records:
        label = str(record.get("label", ""))
        side_record = dict(record.get(side) or {})
        status = str(side_record.get("status", ""))
        structural = side_record.get("structural_grid_xy")
        candidate = side_record.get("candidate_grid_xy")
        used = status == "ok_bilateral" and structural is not None
        rows.append(
            {
                "label": label,
                "source_status": status,
                "structural_grid_xy": structural,
                "candidate_grid_xy": candidate,
                "used_for_fit": bool(used),
                "inlier": None,
                "residual_m": None,
            }
        )
        if used:
            point = np.asarray(structural, dtype=np.float64)
            if point.shape != (2,):
                raise ValueError(f"{label} {side} structural_grid_xy must have 2 values")
            candidate_points.append(point)
            candidate_labels.append(label)

    candidate_count = len(candidate_points)
    if candidate_count < int(min_inlier_count):
        return {
            "fit_status": "insufficient_candidates",
            "candidate_count": candidate_count,
            "inlier_count": 0,
            "outlier_count": 0,
            "fit": None,
            "initial_robust_fit": None,
            "residual_gate_m": None,
            "rows": rows,
        }

    points = np.stack(candidate_points, axis=0)
    u = points @ row_axis
    v = points @ cross_axis
    slope0, intercept0 = _robust_initial_line(v, u)
    residual0_m = (u - (slope0 * v + intercept0)) * float(resolution_m)
    residual_center_m = float(np.median(residual0_m))
    mad_m = float(np.median(np.abs(residual0_m - residual_center_m)))
    robust_sigma_m = 1.4826 * mad_m
    gate_m = max(float(residual_floor_m), float(mad_scale) * robust_sigma_m)
    inlier_mask = np.abs(residual0_m - residual_center_m) <= gate_m + 1e-12
    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count < int(min_inlier_count):
        return {
            "fit_status": "insufficient_inliers",
            "candidate_count": candidate_count,
            "inlier_count": inlier_count,
            "outlier_count": candidate_count - inlier_count,
            "fit": None,
            "initial_robust_fit": {
                "slope_du_dv": slope0,
                "intercept_u": intercept0,
                "residual_center_m": residual_center_m,
                "mad_m": mad_m,
                "robust_sigma_m": robust_sigma_m,
            },
            "residual_gate_m": float(gate_m),
            "rows": rows,
        }

    inlier_v = v[inlier_mask]
    inlier_u = u[inlier_mask]
    if inlier_count >= 2 and float(np.ptp(inlier_v)) > 1e-9:
        slope, intercept = np.polyfit(inlier_v, inlier_u, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = 0.0
        intercept = float(np.mean(inlier_u))

    residual_m = (u - (slope * v + intercept)) * float(resolution_m)
    by_label = {
        label: (bool(inlier), float(residual))
        for label, inlier, residual in zip(candidate_labels, inlier_mask, residual_m)
    }
    for row in rows:
        if not row["used_for_fit"]:
            continue
        inlier, residual = by_label[row["label"]]
        row["inlier"] = inlier
        row["residual_m"] = residual

    rmse_m = float(np.sqrt(np.mean(residual_m[inlier_mask] ** 2)))
    fit = {
        "slope_du_dv": slope,
        "intercept_u": intercept,
        "residual_rmse_m": rmse_m,
        "residual_median_abs_m": float(np.median(np.abs(residual_m[inlier_mask]))),
    }
    fit_status = "ok" if rmse_m <= float(max_fit_rmse_m) + 1e-12 else "poor_fit_quality"
    return {
        "fit_status": fit_status,
        "candidate_count": candidate_count,
        "inlier_count": inlier_count,
        "outlier_count": candidate_count - inlier_count,
        "fit": fit,
        "initial_robust_fit": {
            "method": "median_pairwise_slope_plus_median_intercept",
            "slope_du_dv": slope0,
            "intercept_u": intercept0,
            "residual_center_m": residual_center_m,
            "mad_m": mad_m,
            "robust_sigma_m": robust_sigma_m,
        },
        "residual_gate_m": float(gate_m),
        "rows": rows,
    }


def fit_structural_endpoint_boundaries(
    endpoint_records,
    *,
    row_axis,
    cross_axis,
    resolution_m,
    residual_floor_m=0.30,
    mad_scale=3.0,
    min_inlier_count=3,
    max_fit_rmse_m=0.50,
):
    """Fit robust common entry/exit lines from bilateral structural endpoints."""
    resolution = float(resolution_m)
    floor_m = float(residual_floor_m)
    scale = float(mad_scale)
    minimum = int(min_inlier_count)
    max_rmse = float(max_fit_rmse_m)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")
    if floor_m < 0.0:
        raise ValueError("residual_floor_m must be >= 0")
    if scale < 0.0:
        raise ValueError("mad_scale must be >= 0")
    if minimum < 2:
        raise ValueError("min_inlier_count must be >= 2")
    if max_rmse <= 0.0:
        raise ValueError("max_fit_rmse_m must be > 0")

    axis = _unit(row_axis)
    cross = _unit(cross_axis)
    if abs(float(axis @ cross)) > 1e-6:
        raise ValueError("row_axis and cross_axis must be orthogonal")

    records = list(endpoint_records)
    result = {
        "schema_version": 2,
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "resolution_m": resolution,
        "parameters": {
            "residual_floor_m": floor_m,
            "mad_scale": scale,
            "min_inlier_count": minimum,
            "max_fit_rmse_m": max_rmse,
        },
        "entry": _fit_side(
            records,
            "entry",
            row_axis=axis,
            cross_axis=cross,
            resolution_m=resolution,
            residual_floor_m=floor_m,
            mad_scale=scale,
            min_inlier_count=minimum,
            max_fit_rmse_m=max_rmse,
        ),
        "exit": _fit_side(
            records,
            "exit",
            row_axis=axis,
            cross_axis=cross,
            resolution_m=resolution,
            residual_floor_m=floor_m,
            mad_scale=scale,
            min_inlier_count=minimum,
            max_fit_rmse_m=max_rmse,
        ),
        "policy": {
            "outliers_deleted": False,
            "ambiguous_rows_used_for_fit": False,
            "poor_fit_promoted": False,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result
