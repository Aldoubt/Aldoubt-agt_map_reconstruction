"""Observation-sufficiency evaluation inside uncertainty-aware structural ROIs.

This layer reuses frozen ground and scan/ray evidence arrays.  It does not
change the navigation map, resolve structural uncertainty, or promote UNKNOWN
to semantic free space.  Its purpose is to partition why UNKNOWN remains
unknown in structurally resolved headland ROIs and to report unresolved strips
separately.
"""

from __future__ import annotations

import numpy as np

from .navigation_export import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


_REQUIRED_MASKS = (
    "entry_conservative_outward",
    "entry_boundary_uncertainty",
    "exit_conservative_outward",
    "exit_boundary_uncertainty",
    "structurally_unresolved_cross",
)


def _validate_inputs(base_map, ground_reference, scan_support_count, masks, ray_support_count):
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
    prepared = {}
    for name in _REQUIRED_MASKS:
        if name not in masks:
            raise ValueError(f"missing uncertainty ROI mask: {name}")
        value = np.asarray(masks[name], dtype=bool)
        if value.shape != base.shape:
            raise ValueError(f"uncertainty ROI mask {name} does not match base_map")
        prepared[name] = value
    return base, ground, scan, prepared, ray


def _roi_stats(base, ground, scan, roi, *, min_repeated_scans, ray=None):
    mask = np.asarray(roi, dtype=bool)
    threshold = int(min_repeated_scans)
    unknown = mask & (base == UNKNOWN_VALUE)
    finite_ground = np.isfinite(ground)
    unknown_ground = unknown & finite_ground
    no_ground = unknown & ~finite_ground
    no_observation = unknown_ground & (scan < 1)
    single_scan = unknown_ground & (scan >= 1) & (scan < threshold)
    repeated_scan = unknown_ground & (scan >= threshold)
    ray_supported = None if ray is None else unknown_ground & (ray >= 1)
    partition = no_ground | no_observation | single_scan | repeated_scan
    return {
        "roi_cell_count": int(np.count_nonzero(mask)),
        "free_cell_count": int(np.count_nonzero(mask & (base == FREE_VALUE))),
        "occupied_cell_count": int(np.count_nonzero(mask & (base == OCCUPIED_VALUE))),
        "unknown_cell_count": int(np.count_nonzero(unknown)),
        "unknown_no_ground_reference_cell_count": int(np.count_nonzero(no_ground)),
        "unknown_ground_reference_no_observation_cell_count": int(np.count_nonzero(no_observation)),
        "unknown_single_scan_support_cell_count": int(np.count_nonzero(single_scan)),
        "unknown_repeated_scan_support_cell_count": int(np.count_nonzero(repeated_scan)),
        "ray_supported_unknown_cell_count": (
            None if ray_supported is None else int(np.count_nonzero(ray_supported))
        ),
        "unknown_partition_cell_count": int(np.count_nonzero(partition)),
    }


def evaluate_uncertainty_roi_observation_sufficiency(
    base_map,
    ground_reference,
    scan_support_count,
    roi_masks,
    *,
    min_repeated_scans=2,
    ray_support_count=None,
):
    """Partition observation sufficiency in fused structural endpoint ROIs."""
    threshold = int(min_repeated_scans)
    if threshold < 2:
        raise ValueError("min_repeated_scans must be >= 2 to distinguish single/repeated support")
    base, ground, scan, masks, ray = _validate_inputs(
        base_map,
        ground_reference,
        scan_support_count,
        roi_masks,
        ray_support_count,
    )

    result = {
        "schema_version": 1,
        "method": "fused_structural_roi_observation_sufficiency",
        "grid_shape_yx": list(base.shape),
        "min_repeated_scans": threshold,
        "entry": {
            "conservative_outward": _roi_stats(
                base,
                ground,
                scan,
                masks["entry_conservative_outward"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
            "boundary_uncertainty": _roi_stats(
                base,
                ground,
                scan,
                masks["entry_boundary_uncertainty"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
        },
        "exit": {
            "conservative_outward": _roi_stats(
                base,
                ground,
                scan,
                masks["exit_conservative_outward"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
            "boundary_uncertainty": _roi_stats(
                base,
                ground,
                scan,
                masks["exit_boundary_uncertainty"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
        },
        "structurally_unresolved_cross": _roi_stats(
            base,
            ground,
            scan,
            masks["structurally_unresolved_cross"],
            min_repeated_scans=threshold,
            ray=ray,
        ),
        "policy": {
            "frozen_evidence_reused": True,
            "structural_roi_recomputed_from_evidence": False,
            "unresolved_cross_strip_promoted_to_resolved": False,
            "evaluation_overlay_only": True,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result
