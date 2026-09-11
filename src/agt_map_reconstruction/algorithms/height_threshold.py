"""Simple height-based ground segmentation baseline."""

from __future__ import annotations

import numpy as np

from .base import SegmentationResult
from .registry import register_algorithm


@register_algorithm(
    "height_threshold",
    description="Global percentile-based height threshold baseline",
    version="1",
)
def segment(points: np.ndarray, config=None) -> SegmentationResult:
    config = dict(config or {})
    threshold = float(config.get("height_threshold", 0.15))
    percentile = float(config.get("ground_percentile", 10.0))

    z = points[:, 2]
    z0 = float(np.percentile(z, percentile))
    ground_mask = z < z0 + threshold

    return SegmentationResult(
        ground_points=points[ground_mask],
        non_ground_points=points[~ground_mask],
        metadata={
            "algorithm": "height_threshold",
            "config": {
                "height_threshold": threshold,
                "ground_percentile": percentile,
            },
            "ground_level": z0,
        },
    )
