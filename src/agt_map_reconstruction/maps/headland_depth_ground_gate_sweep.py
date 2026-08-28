"""Ground-reference confidence-gate sensitivity across finite headland depth bands."""

from __future__ import annotations

import numpy as np


def _validate_thresholds(name, values):
    result = sorted(set(float(value) for value in values))
    if not result or any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite values > 0")
    return result


def _prepare_band_masks(payload, masks, shape):
    prepared = []
    occupied = np.zeros(shape, dtype=bool)
    for side in ("entry", "exit"):
        side_payload = dict(payload.get(side) or {})
        for item in side_payload.get("bands") or []:
            key = str(item.get("mask_key", ""))
            if not key or key not in masks:
                raise ValueError(f"missing finite depth mask: {key}")
            mask = np.asarray(masks[key], dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"depth mask {key} does not match input shape")
            if np.any(mask & occupied):
                raise ValueError(f"finite depth masks overlap at {key}")
            occupied |= mask
            prepared.append((side, item, mask))

    unresolved = np.asarray(
        masks.get("structurally_unresolved_cross", np.zeros(shape, dtype=bool)),
        dtype=bool,
    )
    if unresolved.shape != shape:
        raise ValueError("structurally unresolved mask does not match input shape")
    if np.any(unresolved & occupied):
        raise ValueError("finite depth masks overlap structurally unresolved cross strip")
    return prepared


def _sweep_one_band(
    target,
    distance,
    disagreement,
    *,
    distance_thresholds,
    disagreement_thresholds,
):
    target_count = int(np.count_nonzero(target))
    finite_target = target & np.isfinite(distance) & np.isfinite(disagreement)
    grid = []
    for max_distance in distance_thresholds:
        for max_disagreement in disagreement_thresholds:
            accepted = (
                finite_target
                & (distance <= max_distance + 1e-12)
                & (disagreement <= max_disagreement + 1e-12)
            )
            count = int(np.count_nonzero(accepted))
            grid.append(
                {
                    "max_support_distance_m": float(max_distance),
                    "max_model_disagreement_m": float(max_disagreement),
                    "accepted_unknown_cell_count": count,
                    "accepted_unknown_fraction": (
                        0.0 if target_count == 0 else float(count / target_count)
                    ),
                }
            )
    return target_count, int(np.count_nonzero(finite_target)), grid


def sweep_headland_depth_ground_gate(
    unknown_mask,
    depth_profile_payload,
    depth_masks,
    nearest_support_distance_m,
    model_disagreement_m,
    *,
    max_support_distances_m,
    max_model_disagreements_m,
):
    """Sweep joint K8/K16 confidence gates independently in every finite depth band."""
    unknown = np.asarray(unknown_mask, dtype=bool)
    distance = np.asarray(nearest_support_distance_m, dtype=float)
    disagreement = np.asarray(model_disagreement_m, dtype=float)
    if unknown.ndim != 2 or distance.ndim != 2 or disagreement.ndim != 2:
        raise ValueError("unknown, distance, and disagreement grids must be 2D")
    if unknown.shape != distance.shape or unknown.shape != disagreement.shape:
        raise ValueError("unknown, distance, and disagreement grids must match")

    payload = dict(depth_profile_payload)
    expected_shape = tuple(int(value) for value in payload.get("grid_shape_yx", unknown.shape))
    if expected_shape != unknown.shape:
        raise ValueError("depth profile grid shape does not match confidence grids")
    prepared = _prepare_band_masks(payload, depth_masks, unknown.shape)
    distance_thresholds = _validate_thresholds(
        "max_support_distances_m", max_support_distances_m
    )
    disagreement_thresholds = _validate_thresholds(
        "max_model_disagreements_m", max_model_disagreements_m
    )

    sides = {"entry": {"bands": []}, "exit": {"bands": []}}
    for side, item, mask in prepared:
        target = mask & unknown
        target_count, finite_count, grid = _sweep_one_band(
            target,
            distance,
            disagreement,
            distance_thresholds=distance_thresholds,
            disagreement_thresholds=disagreement_thresholds,
        )
        depth_min = float(item.get("depth_min_m"))
        depth_max = float(item.get("depth_max_m"))
        sides[side]["bands"].append(
            {
                "mask_key": str(item.get("mask_key")),
                "depth_min_m": depth_min,
                "depth_max_m": depth_max,
                "depth_midpoint_m": 0.5 * (depth_min + depth_max),
                "unknown_cell_count": target_count,
                "finite_confidence_input_cell_count": finite_count,
                "finite_confidence_input_fraction": (
                    0.0 if target_count == 0 else float(finite_count / target_count)
                ),
                "grid": grid,
            }
        )

    return {
        "schema_version": 1,
        "method": "finite_headland_depth_ground_reference_gate_sweep",
        "grid_shape_yx": list(unknown.shape),
        "support_distance_thresholds_m": distance_thresholds,
        "model_disagreement_thresholds_m": disagreement_thresholds,
        "entry": sides["entry"],
        "exit": sides["exit"],
        "automatic_threshold_selection": False,
        "physical_site_boundary_required": False,
        "structural_geometry_modified": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
