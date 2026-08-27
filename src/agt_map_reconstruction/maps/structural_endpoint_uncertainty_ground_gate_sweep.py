"""Ground-reference confidence-gate sweep inside frozen D3.1 uncertainty ROIs.

The sweep reuses existing nearest-support-distance and local-model-disagreement
grids.  It does not refit ground, alter structural ROI geometry, select a gate
automatically, modify the navigation map, or promote semantic free space.
"""

from __future__ import annotations

import numpy as np


_REGION_NAMES = (
    "entry_conservative_outward",
    "entry_boundary_uncertainty",
    "exit_conservative_outward",
    "exit_boundary_uncertainty",
    "structurally_unresolved_cross",
)


def _validate_values(name, values):
    result = [float(value) for value in values]
    if not result or any((not np.isfinite(value) or value <= 0.0) for value in result):
        raise ValueError(f"{name} must contain finite values > 0")
    return sorted(set(result))


def _prepare_masks(shape, roi_masks):
    prepared = {}
    occupied = np.zeros(shape, dtype=np.uint8)
    for name in _REGION_NAMES:
        if name not in roi_masks:
            raise ValueError(f"missing frozen ROI mask: {name}")
        mask = np.asarray(roi_masks[name], dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"frozen ROI mask {name} does not match input shape")
        if np.any(mask & (occupied > 0)):
            raise ValueError(f"frozen ROI masks overlap at region {name}")
        occupied[mask] = 1
        prepared[name] = mask
    return prepared


def sweep_structural_endpoint_uncertainty_ground_gate(
    unknown_mask,
    roi_masks,
    nearest_support_distance_m,
    model_disagreement_m,
    *,
    max_support_distances_m,
    max_model_disagreements_m,
):
    """Sweep joint ground-confidence gates in the frozen uncertainty ROI partition."""
    unknown = np.asarray(unknown_mask, dtype=bool)
    distance = np.asarray(nearest_support_distance_m, dtype=float)
    disagreement = np.asarray(model_disagreement_m, dtype=float)
    if unknown.ndim != 2 or distance.ndim != 2 or disagreement.ndim != 2:
        raise ValueError("unknown, distance, and disagreement grids must be 2D")
    if unknown.shape != distance.shape or unknown.shape != disagreement.shape:
        raise ValueError("unknown, distance, and disagreement grids must match")

    masks = _prepare_masks(unknown.shape, roi_masks)
    distance_thresholds = _validate_values(
        "max_support_distances_m",
        max_support_distances_m,
    )
    disagreement_thresholds = _validate_values(
        "max_model_disagreements_m",
        max_model_disagreements_m,
    )
    finite = np.isfinite(distance) & np.isfinite(disagreement)

    regions = {}
    for name in _REGION_NAMES:
        target = masks[name] & unknown
        target_count = int(np.count_nonzero(target))
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
                grid.append(
                    {
                        "max_support_distance_m": max_distance,
                        "max_model_disagreement_m": max_disagreement,
                        "accepted_unknown_cell_count": accepted_count,
                        "accepted_unknown_fraction": (
                            0.0
                            if target_count == 0
                            else float(accepted_count / target_count)
                        ),
                    }
                )
        regions[name] = {
            "unknown_cell_count": target_count,
            "finite_confidence_input_fraction": (
                0.0
                if target_count == 0
                else float(np.count_nonzero(finite_target) / target_count)
            ),
            "grid": grid,
        }

    return {
        "schema_version": 1,
        "method": "fused_structural_roi_ground_reference_gate_sweep",
        "support_distance_thresholds_m": distance_thresholds,
        "model_disagreement_thresholds_m": disagreement_thresholds,
        "regions": regions,
        "automatic_threshold_selection": False,
        "structural_roi_modified": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
