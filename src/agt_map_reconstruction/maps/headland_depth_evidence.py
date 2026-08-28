"""Evaluate frozen observation sufficiency by finite headland depth band."""

from __future__ import annotations

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE
from .structural_endpoint_uncertainty_evidence import (
    summarize_observation_sufficiency_roi,
)


def _prepare_inputs(base_map, ground_reference, scan_support_count, ray_support_count):
    base = np.asarray(base_map, dtype=np.uint8)
    ground = np.asarray(ground_reference, dtype=np.float64)
    scan = np.asarray(scan_support_count)
    ray = None if ray_support_count is None else np.asarray(ray_support_count)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if ground.shape != base.shape or scan.shape != base.shape:
        raise ValueError("ground_reference and scan_support_count must match base_map")
    if ray is not None and ray.shape != base.shape:
        raise ValueError("ray_support_count must match base_map")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    return base, ground, scan, ray


def _require_mask(depth_masks, key, shape):
    if key not in depth_masks:
        raise ValueError(f"missing depth-profile mask: {key}")
    mask = np.asarray(depth_masks[key], dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"depth-profile mask {key} does not match base map")
    return mask


def _band_summary(
    base,
    ground,
    scan,
    ray,
    masks,
    item,
    *,
    min_repeated_scans,
):
    key = str(item.get("mask_key", ""))
    if not key:
        raise ValueError("depth band is missing mask_key")
    depth_min = float(item.get("depth_min_m"))
    depth_max = float(item.get("depth_max_m"))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        raise ValueError(f"depth band {key} has invalid numeric bounds")
    mask = _require_mask(masks, key, base.shape)
    stats = summarize_observation_sufficiency_roi(
        base,
        ground,
        scan,
        mask,
        min_repeated_scans=min_repeated_scans,
        ray=ray,
    )
    return {
        "mask_key": key,
        "depth_min_m": depth_min,
        "depth_max_m": depth_max,
        "depth_midpoint_m": 0.5 * (depth_min + depth_max),
        **stats,
    }


def evaluate_headland_depth_evidence(
    base_map,
    ground_reference,
    scan_support_count,
    depth_profile_payload,
    depth_masks,
    *,
    min_repeated_scans=2,
    ray_support_count=None,
):
    """Apply frozen ground/scan/ray evidence to every finite depth band."""
    threshold = int(min_repeated_scans)
    if threshold < 2:
        raise ValueError("min_repeated_scans must be >= 2")
    base, ground, scan, ray = _prepare_inputs(
        base_map,
        ground_reference,
        scan_support_count,
        ray_support_count,
    )
    payload = dict(depth_profile_payload)
    expected_shape = tuple(int(v) for v in payload.get("grid_shape_yx", base.shape))
    if expected_shape != base.shape:
        raise ValueError("depth profile grid shape does not match base map")

    sides = {}
    boundary = {}
    for side in ("entry", "exit"):
        side_payload = dict(payload.get(side) or {})
        bands = [
            _band_summary(
                base,
                ground,
                scan,
                ray,
                depth_masks,
                item,
                min_repeated_scans=threshold,
            )
            for item in (side_payload.get("bands") or [])
        ]
        boundary_key = str(
            side_payload.get("boundary_uncertainty_mask_key")
            or f"{side}_boundary_uncertainty"
        )
        boundary_mask = _require_mask(depth_masks, boundary_key, base.shape)
        boundary[side] = {
            "mask_key": boundary_key,
            **summarize_observation_sufficiency_roi(
                base,
                ground,
                scan,
                boundary_mask,
                min_repeated_scans=threshold,
                ray=ray,
            ),
        }
        sides[side] = {"bands": bands}

    unresolved_key = "structurally_unresolved_cross"
    unresolved = _require_mask(depth_masks, unresolved_key, base.shape)
    unresolved_stats = summarize_observation_sufficiency_roi(
        base,
        ground,
        scan,
        unresolved,
        min_repeated_scans=threshold,
        ray=ray,
    )

    return {
        "schema_version": 1,
        "method": "finite_headland_depth_observation_sufficiency",
        "grid_shape_yx": list(base.shape),
        "depth_edges_m": [float(v) for v in payload.get("depth_edges_m", [])],
        "min_repeated_scans": threshold,
        "entry": sides["entry"],
        "exit": sides["exit"],
        "boundary_uncertainty": boundary,
        "structurally_unresolved_cross": {
            "mask_key": unresolved_key,
            **unresolved_stats,
        },
        "policy": {
            "frozen_evidence_reused": True,
            "rosbag_replay_performed": False,
            "ray_evidence_regenerated": False,
            "physical_site_boundary_required": False,
            "ground_reference_ceiling_is_semantic_free": False,
            "ground_reference_ceiling_is_navigation_acceptance": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
