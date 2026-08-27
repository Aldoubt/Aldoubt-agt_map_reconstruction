"""Chunk-safe accumulation for ground-aware ray evidence.

The global support count is accumulated before applying ``min_support_rays``.
This preserves repeated observations that arrive in different streaming chunks.
"""

from __future__ import annotations

import numpy as np

from .ground_aware_ray_evidence import (
    GroundAwareRayConfig,
    accumulate_ground_aware_ray_support,
)


def accumulate_ground_aware_ray_batches(batches, ground_surface, metadata, config):
    """Accumulate support from an iterable of ObservationRayBundle batches.

    Each batch is evaluated with a temporary support threshold of one. Counts are
    summed globally and ``config.min_support_rays`` is applied only once after all
    batches have been consumed.
    """
    if not isinstance(config, GroundAwareRayConfig):
        raise TypeError("config must be a GroundAwareRayConfig")

    expected_shape = (int(metadata.height), int(metadata.width))
    support = np.zeros(expected_shape, dtype=np.uint64)
    batch_count = 0
    input_ray_count = 0
    accepted_ray_count = 0
    traversed_cell_visits = 0
    supported_cell_visits = 0

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
        summary = result["summary"]
        batch_count += 1
        input_ray_count += int(summary["input_ray_count"])
        accepted_ray_count += int(summary["accepted_ray_count"])
        traversed_cell_visits += int(summary["traversed_cell_visits"])
        supported_cell_visits += int(summary["supported_cell_visits"])

    if batch_count == 0:
        raise ValueError("at least one ray batch is required")

    support_mask = support >= int(config.min_support_rays)
    ground = np.asarray(ground_surface)
    return {
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
            "semantic_promotion": False,
        },
    }
