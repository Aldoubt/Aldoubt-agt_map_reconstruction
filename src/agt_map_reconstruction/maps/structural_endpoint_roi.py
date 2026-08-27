"""D3.1 structural-endpoint ROI reconstruction and frozen-evidence evaluation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def _geometry(boundary_payload, side):
    if side not in {"entry", "exit"}:
        raise ValueError("side must be entry or exit")
    row_axis = _unit(boundary_payload["row_axis_direction"])
    cross_axis = _unit(boundary_payload["cross_row_direction"])
    if abs(float(row_axis @ cross_axis)) > 1e-6:
        raise ValueError("row/cross axes must be orthogonal")
    span = boundary_payload["row_cross_span"]
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError("row_cross_span must contain two values")
    side_fit = boundary_payload["robust_boundary"][side]
    if side_fit.get("fit_status") != "ok" or side_fit.get("fit") is None:
        raise ValueError(f"structural {side} boundary fit is not available")
    fit = side_fit["fit"]
    return row_axis, cross_axis, float(span[0]), float(span[1]), fit


def build_structural_endpoint_roi(shape, boundary_payload, side):
    """Return the frozen cross-row span outward from one D3.1 structural line."""
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")

    row_axis, cross_axis, v_min, v_max, fit = _geometry(boundary_payload, side)
    yy, xx = np.indices((height, width))
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    u = points @ row_axis
    v = points @ cross_axis
    boundary_u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
    cross_roi = (v >= v_min - 1e-12) & (v <= v_max + 1e-12)
    if side == "entry":
        outward = u < boundary_u - 1e-12
    else:
        outward = u > boundary_u + 1e-12
    return (cross_roi & outward).reshape((height, width))


def build_repeated_scan_evaluation_overlay(
    base_map,
    roi,
    ground_reference,
    scan_support_count,
    *,
    min_repeated_scans,
):
    """Build an evaluation-only overlay from repeated, ground-gated scan support."""
    base = np.asarray(base_map, dtype=np.uint8)
    roi_mask = np.asarray(roi, dtype=bool)
    ground = np.asarray(ground_reference, dtype=np.float64)
    support = np.asarray(scan_support_count)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if roi_mask.shape != base.shape or ground.shape != base.shape or support.shape != base.shape:
        raise ValueError("overlay inputs must match base_map shape")
    threshold = int(min_repeated_scans)
    if threshold < 1:
        raise ValueError("min_repeated_scans must be >= 1")

    supported = roi_mask & (support >= threshold)
    finite_ground = np.isfinite(ground)
    promotable = supported & finite_ground & (base == UNKNOWN_VALUE)
    occupied_ignored = supported & (base == OCCUPIED_VALUE)
    existing_free = supported & (base == FREE_VALUE)

    overlay = base.copy()
    overlay[promotable] = FREE_VALUE
    return overlay, {
        "min_repeated_scans": threshold,
        "supported_cell_count": int(np.count_nonzero(supported)),
        "promoted_unknown_cell_count": int(np.count_nonzero(promotable)),
        "supported_existing_free_cell_count": int(np.count_nonzero(existing_free)),
        "supported_occupied_cell_count_ignored": int(np.count_nonzero(occupied_ignored)),
        "evaluation_overlay_only": True,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }


def _strict_safe(base, resolution_m, radius_m):
    free = np.asarray(base, dtype=np.uint8) == FREE_VALUE
    distance = ndimage.distance_transform_edt(free) * float(resolution_m)
    return free & (distance + 1e-12 >= float(radius_m))


def _endpoint_points(boundary_payload, side):
    points = []
    for row in boundary_payload.get("rows", []):
        record = row.get(side) or {}
        if record.get("status") != "ok_bilateral":
            continue
        point = record.get("structural_grid_xy")
        if point is None:
            continue
        value = np.asarray(point, dtype=np.float64)
        if value.shape == (2,):
            points.append(value)
    return points


def _best_component_metrics(base, roi, boundary_payload, side, *, radius_m):
    resolution = float(boundary_payload["resolution_m"])
    safe = _strict_safe(base, resolution, radius_m) & np.asarray(roi, dtype=bool)
    labels, count = ndimage.label(safe)
    row_axis, cross_axis, v_min, v_max, fit = _geometry(boundary_payload, side)
    endpoints = _endpoint_points(boundary_payload, side)
    span = max(1e-12, v_max - v_min)
    components = []

    for component_id in range(1, int(count) + 1):
        yy, xx = np.nonzero(labels == component_id)
        if xx.size == 0:
            continue
        points = np.column_stack((xx.astype(float), yy.astype(float)))
        u = points @ row_axis
        v = points @ cross_axis
        boundary_u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
        if side == "entry":
            depth = boundary_u - u
        else:
            depth = u - boundary_u
        overlap = max(0.0, min(float(np.max(v)), v_max) - max(float(np.min(v)), v_min))

        endpoint_distances = []
        if endpoints:
            tree = cKDTree(points)
            distances, _ = tree.query(np.stack(endpoints, axis=0), k=1)
            endpoint_distances = (np.asarray(distances) * resolution).tolist()
        finite_distances = [value for value in endpoint_distances if np.isfinite(value)]
        components.append(
            {
                "component_id": int(component_id),
                "cell_count": int(points.shape[0]),
                "area_m2": float(points.shape[0]) * resolution * resolution,
                "cross_row_coverage_fraction": float(np.clip(overlap / span, 0.0, 1.0)),
                "endpoint_distance_median_m": (
                    float(np.median(finite_distances)) if finite_distances else None
                ),
                "max_outward_depth_m": float(np.max(depth)) * resolution,
            }
        )

    best = max(
        components,
        key=lambda item: (
            item["cross_row_coverage_fraction"],
            -float("inf")
            if item["endpoint_distance_median_m"] is None
            else -item["endpoint_distance_median_m"],
            item["max_outward_depth_m"],
            item["cell_count"],
        ),
        default=None,
    )
    return {
        "component_count": len(components),
        "best_component": best,
        "components": components,
    }


def _metric_delta(baseline, candidate):
    if baseline is None or candidate is None:
        return {
            "coverage_gain": None,
            "endpoint_distance_reduction_m": None,
            "outward_depth_gain_m": None,
        }
    bdist = baseline.get("endpoint_distance_median_m")
    cdist = candidate.get("endpoint_distance_median_m")
    return {
        "coverage_gain": float(
            candidate["cross_row_coverage_fraction"]
            - baseline["cross_row_coverage_fraction"]
        ),
        "endpoint_distance_reduction_m": (
            None if bdist is None or cdist is None else float(bdist - cdist)
        ),
        "outward_depth_gain_m": float(
            candidate["max_outward_depth_m"] - baseline["max_outward_depth_m"]
        ),
    }


def evaluate_structural_endpoint_evidence(
    base_map,
    boundary_payload,
    *,
    ground_reference,
    scan_support_count,
    ray_support_count=None,
    min_repeated_scans=2,
    radius_m=None,
):
    """Re-evaluate frozen observation evidence using only the D3.1 ROI geometry."""
    base = np.asarray(base_map, dtype=np.uint8)
    ground = np.asarray(ground_reference, dtype=np.float64)
    scan = np.asarray(scan_support_count)
    ray = None if ray_support_count is None else np.asarray(ray_support_count)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if ground.shape != base.shape or scan.shape != base.shape:
        raise ValueError("ground/scan arrays must match base_map")
    if ray is not None and ray.shape != base.shape:
        raise ValueError("ray_support_count must match base_map")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")

    resolution = float(boundary_payload["resolution_m"])
    radius = (
        float(radius_m)
        if radius_m is not None
        else float(boundary_payload.get("radius_m", 0.20))
    )
    if radius < 0.0:
        raise ValueError("radius_m must be >= 0")

    sides = {}
    for side in ("entry", "exit"):
        roi = build_structural_endpoint_roi(base.shape, boundary_payload, side)
        unknown = roi & (base == UNKNOWN_VALUE)
        finite_ground_unknown = unknown & np.isfinite(ground)
        repeated_unknown = finite_ground_unknown & (scan >= int(min_repeated_scans))
        evidence = {
            "roi_cell_count": int(np.count_nonzero(roi)),
            "unknown_cell_count": int(np.count_nonzero(unknown)),
            "ground_finite_unknown_cell_count": int(np.count_nonzero(finite_ground_unknown)),
            "repeated_scan_supported_unknown_cell_count": int(
                np.count_nonzero(repeated_unknown)
            ),
            "single_or_more_scan_supported_unknown_cell_count": int(
                np.count_nonzero(finite_ground_unknown & (scan >= 1))
            ),
            "ray_supported_unknown_cell_count": (
                None
                if ray is None
                else int(np.count_nonzero(finite_ground_unknown & (ray >= 1)))
            ),
        }

        baseline_metrics = _best_component_metrics(
            base,
            roi,
            boundary_payload,
            side,
            radius_m=radius,
        )
        overlay, overlay_summary = build_repeated_scan_evaluation_overlay(
            base,
            roi,
            ground,
            scan,
            min_repeated_scans=int(min_repeated_scans),
        )
        candidate_metrics = _best_component_metrics(
            overlay,
            roi,
            boundary_payload,
            side,
            radius_m=radius,
        )
        sides[side] = {
            "evidence": evidence,
            "baseline_strict": baseline_metrics,
            "candidate_repeated_scan_overlay": candidate_metrics,
            "candidate_overlay": overlay_summary,
            "delta": _metric_delta(
                baseline_metrics.get("best_component"),
                candidate_metrics.get("best_component"),
            ),
        }

    return {
        "schema_version": 1,
        "resolution_m": resolution,
        "radius_m": radius,
        "min_repeated_scans": int(min_repeated_scans),
        "entry": sides["entry"],
        "exit": sides["exit"],
        "policy": {
            "frozen_evidence_reused": True,
            "evaluation_overlay_only": True,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
