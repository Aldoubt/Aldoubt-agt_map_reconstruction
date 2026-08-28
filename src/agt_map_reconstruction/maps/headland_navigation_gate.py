"""Explicit evidence gate for conservative headland free-space promotion.

This module converts already-frozen headland depth geometry and observation
confidence grids into a boolean trusted-free mask.  It never chooses thresholds
automatically and never edits a navigation map itself.
"""

from __future__ import annotations

import numpy as np

from .navigation_export import UNKNOWN_VALUE


def _require_grid(name, value, shape, *, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"{name} must be a 2D grid matching base_map")
    return array


def _require_mask(masks, key, shape):
    if key not in masks:
        raise ValueError(f"missing headland depth mask: {key}")
    return _require_grid(key, masks[key], shape, dtype=bool)


def _validate_nonnegative(name, value):
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


def _validate_positive(name, value):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def build_headland_navigation_gate(
    base_map,
    depth_profile_payload,
    depth_masks,
    nearest_support_distance_m,
    model_disagreement_m,
    scan_support_count,
    *,
    entry_max_depth_m,
    exit_max_depth_m,
    max_support_distance_m,
    max_model_disagreement_m,
    min_scan_support=1,
    ray_support_count=None,
    min_ray_support=0,
):
    """Build trusted-free and uncertainty masks using explicit fixed thresholds.

    Only cells that are UNKNOWN in ``base_map`` may be promoted.  A whole depth
    band is eligible only when its upper edge is within the configured maximum
    depth for that side.  Structural boundary-uncertainty masks and unresolved
    cross-row strips are always excluded from trusted free space and returned as
    a separate uncertainty mask for downstream conservative export.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    shape = base.shape

    payload = dict(depth_profile_payload)
    expected_shape = tuple(int(v) for v in payload.get("grid_shape_yx", shape))
    if expected_shape != shape:
        raise ValueError("depth profile grid shape does not match base_map")

    distance = _require_grid(
        "nearest_support_distance_m", nearest_support_distance_m, shape, dtype=float
    )
    disagreement = _require_grid(
        "model_disagreement_m", model_disagreement_m, shape, dtype=float
    )
    scans = _require_grid("scan_support_count", scan_support_count, shape)
    rays = None
    if ray_support_count is not None:
        rays = _require_grid("ray_support_count", ray_support_count, shape)

    entry_depth = _validate_nonnegative("entry_max_depth_m", entry_max_depth_m)
    exit_depth = _validate_nonnegative("exit_max_depth_m", exit_max_depth_m)
    support_limit = _validate_positive(
        "max_support_distance_m", max_support_distance_m
    )
    disagreement_limit = _validate_positive(
        "max_model_disagreement_m", max_model_disagreement_m
    )
    min_scans = int(min_scan_support)
    min_rays = int(min_ray_support)
    if min_scans < 0 or min_rays < 0:
        raise ValueError("minimum scan/ray support must be >= 0")
    if min_rays > 0 and rays is None:
        raise ValueError("ray_support_count is required when min_ray_support > 0")

    uncertainty = np.zeros(shape, dtype=bool)
    for side in ("entry", "exit"):
        side_payload = dict(payload.get(side) or {})
        boundary_key = str(
            side_payload.get("boundary_uncertainty_mask_key")
            or f"{side}_boundary_uncertainty"
        )
        uncertainty |= _require_mask(depth_masks, boundary_key, shape)
    uncertainty |= _require_mask(
        depth_masks, "structurally_unresolved_cross", shape
    )

    unknown = base == UNKNOWN_VALUE
    finite_confidence = np.isfinite(distance) & np.isfinite(disagreement)
    confidence_ok = (
        finite_confidence
        & (distance <= support_limit + 1e-12)
        & (disagreement <= disagreement_limit + 1e-12)
        & (scans >= min_scans)
    )
    if min_rays > 0:
        confidence_ok &= rays >= min_rays

    trusted = np.zeros(shape, dtype=bool)
    side_results = {}
    for side, max_depth in (("entry", entry_depth), ("exit", exit_depth)):
        bands = []
        for item in (dict(payload.get(side) or {}).get("bands") or []):
            key = str(item.get("mask_key", ""))
            if not key:
                raise ValueError(f"{side} depth band is missing mask_key")
            depth_min = float(item.get("depth_min_m"))
            depth_max = float(item.get("depth_max_m"))
            if (
                not np.isfinite(depth_min)
                or not np.isfinite(depth_max)
                or depth_min < 0.0
                or depth_max <= depth_min
            ):
                raise ValueError(f"depth band {key} has invalid bounds")
            mask = _require_mask(depth_masks, key, shape)
            band_enabled = max_depth > 0.0 and depth_max <= max_depth + 1e-12
            selected = (
                mask & unknown & confidence_ok & ~uncertainty
                if band_enabled
                else np.zeros(shape, dtype=bool)
            )
            trusted |= selected
            target = mask & unknown
            bands.append(
                {
                    "mask_key": key,
                    "depth_min_m": depth_min,
                    "depth_max_m": depth_max,
                    "band_enabled": bool(band_enabled),
                    "unknown_cell_count": int(np.count_nonzero(target)),
                    "trusted_free_cell_count": int(np.count_nonzero(selected)),
                    "trusted_free_fraction_of_unknown": (
                        0.0
                        if not np.any(target)
                        else float(np.count_nonzero(selected) / np.count_nonzero(target))
                    ),
                }
            )
        side_results[side] = {
            "max_depth_m": float(max_depth),
            "bands": bands,
            "trusted_free_cell_count": int(
                sum(item["trusted_free_cell_count"] for item in bands)
            ),
        }

    result = {
        "schema_version": 1,
        "method": "explicit_conservative_headland_navigation_gate",
        "grid_shape_yx": list(shape),
        "entry": side_results["entry"],
        "exit": side_results["exit"],
        "max_support_distance_m": float(support_limit),
        "max_model_disagreement_m": float(disagreement_limit),
        "min_scan_support": min_scans,
        "min_ray_support": min_rays,
        "trusted_free_cell_count": int(np.count_nonzero(trusted)),
        "uncertainty_cell_count": int(np.count_nonzero(uncertainty)),
        "automatic_threshold_selection": False,
        "automatic_depth_selection": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
        "policy": {
            "unknown_only_promotion": True,
            "whole_depth_bands_only": True,
            "boundary_uncertainty_excluded": True,
            "structurally_unresolved_cross_excluded": True,
            "thresholds_are_explicit_inputs": True,
        },
    }
    return result, trusted, uncertainty
