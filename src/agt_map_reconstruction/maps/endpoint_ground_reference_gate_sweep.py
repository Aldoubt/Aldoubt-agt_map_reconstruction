"""Sweep support-distance and model-disagreement gates inside frozen D3 endpoint ROIs."""

from __future__ import annotations

import numpy as np


def _roi_masks(shape, envelope):
    row_axis = np.asarray(envelope["row_axis_direction"], dtype=float)
    cross_axis = np.asarray(envelope["cross_row_direction"], dtype=float)
    row_v_min, row_v_max = map(float, envelope["row_cross_span"])

    yy, xx = np.indices(shape)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(float)
    u = points @ row_axis
    v = points @ cross_axis
    cross_roi = (v >= row_v_min - 1e-12) & (v <= row_v_max + 1e-12)

    masks = {}
    for side in ("entry", "exit"):
        fit = envelope["sides"][side]["endpoint_fit"]
        boundary_u = float(fit["slope_du_dv"]) * v + float(fit["intercept_u"])
        outward = u < boundary_u - 1e-12 if side == "entry" else u > boundary_u + 1e-12
        masks[side] = (cross_roi & outward).reshape(shape)
    return masks


def _validate_values(name, values):
    result = [float(v) for v in values]
    if not result or any((not np.isfinite(v) or v <= 0.0) for v in result):
        raise ValueError(f"{name} must contain finite values > 0")
    return sorted(set(result))


def sweep_endpoint_ground_reference_gate(
    unknown_mask,
    endpoint_envelope,
    nearest_support_distance_m,
    model_disagreement_m,
    max_support_distances_m,
    max_model_disagreements_m,
):
    """Return joint gate acceptance fractions for unknown cells in D3 endpoint ROIs."""
    unknown = np.asarray(unknown_mask, dtype=bool)
    distance = np.asarray(nearest_support_distance_m, dtype=float)
    disagreement = np.asarray(model_disagreement_m, dtype=float)
    if unknown.ndim != 2 or distance.ndim != 2 or disagreement.ndim != 2:
        raise ValueError("unknown, distance, and disagreement grids must be 2D")
    if unknown.shape != distance.shape or unknown.shape != disagreement.shape:
        raise ValueError("unknown, distance, and disagreement grids must match")

    distance_thresholds = _validate_values(
        "max_support_distances_m", max_support_distances_m
    )
    disagreement_thresholds = _validate_values(
        "max_model_disagreements_m", max_model_disagreements_m
    )
    finite = np.isfinite(distance) & np.isfinite(disagreement)
    rois = _roi_masks(unknown.shape, endpoint_envelope)

    sides = {}
    for side, roi in rois.items():
        target = roi & unknown
        count = int(np.count_nonzero(target))
        finite_target = target & finite
        grid = []
        for max_distance in distance_thresholds:
            for max_disagreement in disagreement_thresholds:
                accepted = (
                    finite_target
                    & (distance <= max_distance + 1e-12)
                    & (disagreement <= max_disagreement + 1e-12)
                )
                accepted_count = int(np.count_nonzero(accepted))
                grid.append({
                    "max_support_distance_m": max_distance,
                    "max_model_disagreement_m": max_disagreement,
                    "accepted_unknown_cell_count": accepted_count,
                    "accepted_unknown_fraction": (
                        0.0 if count == 0 else float(accepted_count / count)
                    ),
                })
        sides[side] = {
            "unknown_cell_count": count,
            "finite_confidence_input_fraction": (
                0.0 if count == 0 else float(np.count_nonzero(finite_target) / count)
            ),
            "grid": grid,
        }

    return {
        "schema_version": 1,
        "support_distance_thresholds_m": distance_thresholds,
        "model_disagreement_thresholds_m": disagreement_thresholds,
        "sides": sides,
        "automatic_threshold_selection": False,
        "semantic_promotion": False,
    }
