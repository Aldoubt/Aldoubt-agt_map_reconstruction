"""Audit ground-reference confidence inside the frozen P1-D3 endpoint ROIs."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .navigation_export import UNKNOWN_VALUE


def _unit(vector, name):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError(f"{name} must contain two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "sample_count": 0,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
        }
    return {
        "sample_count": int(values.size),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95.0)),
        "max_m": float(np.max(values)),
    }


def _endpoint_roi_masks(shape, endpoint_envelope):
    if len(shape) != 2:
        raise ValueError("map shape must be two-dimensional")
    row_axis = _unit(endpoint_envelope.get("row_axis_direction"), "row_axis_direction")
    cross_axis = _unit(
        endpoint_envelope.get("cross_row_direction"), "cross_row_direction"
    )
    span = np.asarray(endpoint_envelope.get("row_cross_span"), dtype=np.float64).reshape(-1)
    if span.size != 2 or not np.all(np.isfinite(span)) or span[1] < span[0]:
        raise ValueError("row_cross_span must contain finite [min, max]")

    yy, xx = np.indices(shape)
    u = xx.astype(np.float64) * row_axis[0] + yy.astype(np.float64) * row_axis[1]
    v = xx.astype(np.float64) * cross_axis[0] + yy.astype(np.float64) * cross_axis[1]
    cross_roi = (v >= span[0] - 1e-12) & (v <= span[1] + 1e-12)

    sides = endpoint_envelope.get("sides")
    if not isinstance(sides, dict):
        raise ValueError("endpoint envelope must contain sides")

    masks = {}
    for side in ("entry", "exit"):
        side_payload = sides.get(side)
        if not isinstance(side_payload, dict):
            raise ValueError(f"endpoint envelope missing {side} side")
        fit = side_payload.get("endpoint_fit")
        if not isinstance(fit, dict):
            raise ValueError(f"endpoint envelope {side} missing endpoint_fit")
        slope = float(fit["slope_du_dv"])
        intercept = float(fit["intercept_u"])
        boundary_u = slope * v + intercept
        outward = u < boundary_u - 1e-12 if side == "entry" else u > boundary_u + 1e-12
        masks[side] = cross_roi & outward
    return masks


def _validate_model(name, payload, shape):
    if not isinstance(payload, dict):
        raise ValueError(f"model {name} must be a mapping")
    reference = np.asarray(payload.get("reference"), dtype=np.float64)
    valid_mask = np.asarray(payload.get("valid_mask"), dtype=bool)
    if reference.shape != shape:
        raise ValueError(f"model {name} reference shape {reference.shape} != {shape}")
    if valid_mask.shape != shape:
        raise ValueError(f"model {name} valid-mask shape {valid_mask.shape} != {shape}")
    return reference, valid_mask


def audit_endpoint_ground_reference_confidence(
    base_map,
    endpoint_envelope,
    models,
    nearest_support_distance_m,
):
    """Compare local ground references inside the exact frozen P1-D3 endpoint ROIs.

    The audit is descriptive only. It reports unknown-cell extrapolation distance,
    per-model validity, and cross-model height disagreement. No threshold or model
    choice is applied, and no semantic cell is promoted.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if not isinstance(models, dict) or len(models) < 2:
        raise ValueError("at least two ground-reference models are required")

    nearest = np.asarray(nearest_support_distance_m, dtype=np.float64)
    if nearest.shape != base.shape:
        raise ValueError(
            f"nearest-support distance shape {nearest.shape} != map shape {base.shape}"
        )

    prepared = {}
    for name, payload in models.items():
        reference, valid_mask = _validate_model(name, payload, base.shape)
        supplied_nearest = payload.get("nearest_support_distance_m")
        if supplied_nearest is not None:
            supplied_nearest = np.asarray(supplied_nearest, dtype=np.float64)
            if supplied_nearest.shape != base.shape or not np.allclose(
                supplied_nearest,
                nearest,
                rtol=0.0,
                atol=1e-6,
                equal_nan=True,
            ):
                raise ValueError(
                    f"model {name} nearest-support distance differs from canonical grid"
                )
        prepared[str(name)] = {
            "reference": reference,
            "valid": valid_mask & np.isfinite(reference),
            "neighbor_count": payload.get("neighbor_count"),
            "cv_residual_rmse_m": payload.get("cv_residual_rmse_m"),
            "cv_residual_p95_abs_m": payload.get("cv_residual_p95_abs_m"),
        }

    roi_masks = _endpoint_roi_masks(base.shape, endpoint_envelope)
    result_sides = {}
    ordered_names = sorted(
        prepared,
        key=lambda name: (
            float("inf")
            if prepared[name]["neighbor_count"] is None
            else int(prepared[name]["neighbor_count"]),
            name,
        ),
    )

    for side, roi in roi_masks.items():
        unknown = roi & (base == UNKNOWN_VALUE)
        unknown_count = int(np.count_nonzero(unknown))
        distance_stats = _summary(nearest[unknown])

        model_summary = {}
        common_valid = unknown.copy()
        for name in ordered_names:
            model = prepared[name]
            valid_unknown = unknown & model["valid"]
            common_valid &= model["valid"]
            model_summary[name] = {
                "neighbor_count": model["neighbor_count"],
                "cv_residual_rmse_m": model["cv_residual_rmse_m"],
                "cv_residual_p95_abs_m": model["cv_residual_p95_abs_m"],
                "valid_unknown_cell_count": int(np.count_nonzero(valid_unknown)),
                "valid_unknown_fraction": (
                    None
                    if unknown_count == 0
                    else float(np.count_nonzero(valid_unknown) / unknown_count)
                ),
            }

        common_count = int(np.count_nonzero(common_valid))
        if common_count:
            stack = np.stack(
                [prepared[name]["reference"][common_valid] for name in ordered_names],
                axis=0,
            )
            reference_range = np.max(stack, axis=0) - np.min(stack, axis=0)
        else:
            reference_range = np.empty(0, dtype=np.float64)
        range_stats = _summary(reference_range)
        cross_model = {
            "model_names": ordered_names,
            "common_valid_unknown_cell_count": common_count,
            "common_valid_unknown_fraction": (
                None if unknown_count == 0 else float(common_count / unknown_count)
            ),
            "range_median_m": range_stats["median_m"],
            "range_p95_m": range_stats["p95_m"],
            "range_max_m": range_stats["max_m"],
        }

        pairwise = {}
        for left_name, right_name in combinations(ordered_names, 2):
            left = prepared[left_name]
            right = prepared[right_name]
            pair_valid = unknown & left["valid"] & right["valid"]
            difference = np.abs(
                left["reference"][pair_valid] - right["reference"][pair_valid]
            )
            stats = _summary(difference)
            key = f"{left_name}__{right_name}"
            pairwise[key] = {
                "common_valid_unknown_cell_count": int(np.count_nonzero(pair_valid)),
                "median_m": stats["median_m"],
                "p95_m": stats["p95_m"],
                "max_m": stats["max_m"],
            }

        result_sides[side] = {
            "roi_cell_count": int(np.count_nonzero(roi)),
            "unknown_cell_count": unknown_count,
            "unknown_fraction": (
                None
                if not np.count_nonzero(roi)
                else float(unknown_count / np.count_nonzero(roi))
            ),
            "nearest_support_distance_sample_count": distance_stats["sample_count"],
            "nearest_support_distance_median_m": distance_stats["median_m"],
            "nearest_support_distance_p95_m": distance_stats["p95_m"],
            "nearest_support_distance_max_m": distance_stats["max_m"],
            "models": model_summary,
            "cross_model_disagreement": cross_model,
            "pairwise_abs_difference": pairwise,
        }

    return {
        "schema_version": 1,
        "radius_m": endpoint_envelope.get("radius_m"),
        "model_names": ordered_names,
        "policy": {
            "endpoint_roi_source": "frozen_p1_d3_endpoint_envelope",
            "unknown_only_for_extrapolation_audit": True,
            "automatic_model_selection": False,
            "semantic_promotion": False,
        },
        "sides": result_sides,
    }
