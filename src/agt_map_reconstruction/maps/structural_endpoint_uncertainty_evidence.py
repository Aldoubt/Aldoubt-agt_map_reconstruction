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


def _fraction(numerator, denominator):
    denominator = int(denominator)
    if denominator <= 0:
        return None
    return float(int(numerator) / denominator)


def summarize_observation_sufficiency_roi(
    base,
    ground,
    scan,
    roi,
    *,
    min_repeated_scans,
    ray=None,
):
    """Summarize frozen ground/scan/ray evidence inside one evaluation ROI."""
    base = np.asarray(base, dtype=np.uint8)
    ground = np.asarray(ground, dtype=np.float64)
    scan = np.asarray(scan)
    mask = np.asarray(roi, dtype=bool)
    ray_array = None if ray is None else np.asarray(ray)
    if base.ndim != 2:
        raise ValueError("base must be 2D")
    if ground.shape != base.shape or scan.shape != base.shape or mask.shape != base.shape:
        raise ValueError("ground, scan, and roi must match base shape")
    if ray_array is not None and ray_array.shape != base.shape:
        raise ValueError("ray must match base shape")
    threshold = int(min_repeated_scans)
    if threshold < 2:
        raise ValueError("min_repeated_scans must be >= 2 to distinguish single/repeated support")

    unknown = mask & (base == UNKNOWN_VALUE)
    finite_ground = np.isfinite(ground)
    unknown_ground = unknown & finite_ground
    no_ground = unknown & ~finite_ground
    no_observation = unknown_ground & (scan < 1)
    single_scan = unknown_ground & (scan >= 1) & (scan < threshold)
    repeated_scan = unknown_ground & (scan >= threshold)
    scan_observed = single_scan | repeated_scan
    ray_supported = None if ray_array is None else unknown_ground & (ray_array >= 1)
    partition = no_ground | no_observation | single_scan | repeated_scan

    roi_count = int(np.count_nonzero(mask))
    unknown_count = int(np.count_nonzero(unknown))
    trusted_ground_count = int(np.count_nonzero(unknown_ground))
    no_ground_count = int(np.count_nonzero(no_ground))
    no_observation_count = int(np.count_nonzero(no_observation))
    single_scan_count = int(np.count_nonzero(single_scan))
    repeated_scan_count = int(np.count_nonzero(repeated_scan))
    scan_observed_count = int(np.count_nonzero(scan_observed))
    ray_supported_count = (
        None if ray_supported is None else int(np.count_nonzero(ray_supported))
    )

    return {
        "roi_cell_count": roi_count,
        "free_cell_count": int(np.count_nonzero(mask & (base == FREE_VALUE))),
        "occupied_cell_count": int(np.count_nonzero(mask & (base == OCCUPIED_VALUE))),
        "unknown_cell_count": unknown_count,
        "unknown_fraction_of_roi": _fraction(unknown_count, roi_count),
        "trusted_ground_unknown_cell_count": trusted_ground_count,
        "ground_reference_ceiling_fraction_of_unknown": _fraction(
            trusted_ground_count,
            unknown_count,
        ),
        "unknown_no_ground_reference_cell_count": no_ground_count,
        "unknown_no_ground_reference_fraction": _fraction(no_ground_count, unknown_count),
        "unknown_ground_reference_no_observation_cell_count": no_observation_count,
        "ground_reference_no_observation_fraction_of_trusted_ground_unknown": _fraction(
            no_observation_count,
            trusted_ground_count,
        ),
        "unknown_single_scan_support_cell_count": single_scan_count,
        "unknown_repeated_scan_support_cell_count": repeated_scan_count,
        "scan_observed_unknown_cell_count": scan_observed_count,
        "scan_observed_fraction_of_trusted_ground_unknown": _fraction(
            scan_observed_count,
            trusted_ground_count,
        ),
        "scan_observed_fraction_of_unknown": _fraction(
            scan_observed_count,
            unknown_count,
        ),
        "repeated_scan_fraction_of_trusted_ground_unknown": _fraction(
            repeated_scan_count,
            trusted_ground_count,
        ),
        "repeated_scan_fraction_of_unknown": _fraction(
            repeated_scan_count,
            unknown_count,
        ),
        "ray_supported_unknown_cell_count": ray_supported_count,
        "ray_supported_fraction_of_trusted_ground_unknown": (
            None
            if ray_supported_count is None
            else _fraction(ray_supported_count, trusted_ground_count)
        ),
        "ray_supported_fraction_of_unknown": (
            None
            if ray_supported_count is None
            else _fraction(ray_supported_count, unknown_count)
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
        "schema_version": 2,
        "method": "fused_structural_roi_observation_sufficiency",
        "grid_shape_yx": list(base.shape),
        "min_repeated_scans": threshold,
        "entry": {
            "conservative_outward": summarize_observation_sufficiency_roi(
                base,
                ground,
                scan,
                masks["entry_conservative_outward"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
            "boundary_uncertainty": summarize_observation_sufficiency_roi(
                base,
                ground,
                scan,
                masks["entry_boundary_uncertainty"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
        },
        "exit": {
            "conservative_outward": summarize_observation_sufficiency_roi(
                base,
                ground,
                scan,
                masks["exit_conservative_outward"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
            "boundary_uncertainty": summarize_observation_sufficiency_roi(
                base,
                ground,
                scan,
                masks["exit_boundary_uncertainty"],
                min_repeated_scans=threshold,
                ray=ray,
            ),
        },
        "structurally_unresolved_cross": summarize_observation_sufficiency_roi(
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
            "ground_reference_ceiling_is_semantic_free": False,
            "ground_reference_ceiling_is_navigation_acceptance": False,
            "evaluation_overlay_only": True,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result
