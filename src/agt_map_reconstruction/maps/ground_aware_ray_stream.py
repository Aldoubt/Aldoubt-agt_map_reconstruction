"""Chunk-safe accumulation for ground-aware ray evidence.

Ray counts and optional unique-scan counts are accumulated globally before any
support threshold is interpreted. Streaming batches must not split a physical
scan when scan support is requested.
"""

from __future__ import annotations

import numpy as np

from .ground_aware_ray_evidence import (
    GroundAwareRayConfig,
    accumulate_ground_aware_ray_support,
)


def accumulate_ground_aware_ray_batches(batches, ground_surface, metadata, config):
    """Accumulate support from an iterable of ObservationRayBundle batches.

    Each batch is evaluated with a temporary ray threshold of one. Raw ray counts
    are summed globally and ``config.min_support_rays`` is applied only after all
    batches are consumed. When batches carry ``scan_index``, unique physical-scan
    support is accumulated independently of ray density.
    """
    if not isinstance(config, GroundAwareRayConfig):
        raise TypeError("config must be a GroundAwareRayConfig")

    expected_shape = (int(metadata.height), int(metadata.width))
    support = np.zeros(expected_shape, dtype=np.uint64)
    scan_support = np.zeros(expected_shape, dtype=np.uint64)
    scan_support_available = None
    batch_count = 0
    input_ray_count = 0
    accepted_ray_count = 0
    traversed_cell_visits = 0
    supported_cell_visits = 0
    scan_supported_cell_visits = 0

    chunk_config = GroundAwareRayConfig(
        min_ground_relative_height_m=config.min_ground_relative_height_m,
        max_ground_relative_height_m=config.max_ground_relative_height_m,
        min_support_rays=1,
        min_ray_range_m=config.min_ray_range_m,
        max_ray_range_m=config.max_ray_range_m,
    )

    for bundle in batches:
        result = accumulate_ground_aware_ray_support(
            bundle,
            ground_surface,
            metadata,
            chunk_config,
        )
        support += result["support_count"].astype(np.uint64, copy=False)
        has_scan_support = "scan_support_count" in result
        if scan_support_available is None:
            scan_support_available = has_scan_support
        elif bool(scan_support_available) != bool(has_scan_support):
            raise ValueError("all ray batches must consistently include scan_index")
        if has_scan_support:
            scan_support += result["scan_support_count"].astype(np.uint64, copy=False)

        summary = result["summary"]
        batch_count += 1
        input_ray_count += int(summary["input_ray_count"])
        accepted_ray_count += int(summary["accepted_ray_count"])
        traversed_cell_visits += int(summary["traversed_cell_visits"])
        supported_cell_visits += int(summary["supported_cell_visits"])
        scan_supported_cell_visits += int(summary.get("scan_supported_cell_visits", 0))

    if batch_count == 0:
        raise ValueError("at least one ray batch is required")

    support_mask = support >= int(config.min_support_rays)
    ground = np.asarray(ground_surface)
    result = {
        "support_count": support,
        "support_mask": support_mask,
        "summary": {
            "schema_version": 1,
            "batch_count": int(batch_count),
            "input_ray_count": int(input_ray_count),
            "accepted_ray_count": int(accepted_ray_count),
            "traversed_cell_visits": int(traversed_cell_visits),
            "supported_cell_visits": int(supported_cell_visits),
            "supported_cell_count": int(np.count_nonzero(support_mask)),
            "finite_ground_cell_count": int(np.count_nonzero(np.isfinite(ground))),
            "min_support_rays_applied_after_global_accumulation": True,
            "scan_support_available": bool(scan_support_available),
            "scan_supported_cell_visits": int(scan_supported_cell_visits),
            "semantic_promotion": False,
        },
    }
    if scan_support_available:
        result["scan_support_count"] = scan_support
        result["summary"]["scan_supported_cell_count"] = int(np.count_nonzero(scan_support))
        result["summary"]["max_scan_support_count"] = int(np.max(scan_support))
    return result
