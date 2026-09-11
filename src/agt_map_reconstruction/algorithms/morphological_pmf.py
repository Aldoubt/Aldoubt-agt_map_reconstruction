"""Lightweight PMF-inspired segmentation baseline.

This is intentionally not a canonical Progressive Morphological Filter. It is
kept as a low-dependency comparison baseline until a production PMF/PCL adapter
is introduced.
"""

from __future__ import annotations

import numpy as np

from .base import SegmentationResult
from .registry import register_algorithm


@register_algorithm(
    "morphological_pmf",
    description="Histogram-mode PMF-inspired lightweight baseline",
    version="1",
)
def segment(points: np.ndarray, config=None) -> SegmentationResult:
    config = dict(config or {})
    height = float(config.get("height_threshold", 0.25))
    bins = int(config.get("bins", 200))
    if bins < 2:
        raise ValueError("bins must be >= 2")

    z = points[:, 2]
    hist, edges = np.histogram(z, bins=bins)
    ground_level = float(edges[int(np.argmax(hist))])
    mask = z < ground_level + height

    return SegmentationResult(
        ground_points=points[mask],
        non_ground_points=points[~mask],
        metadata={
            "algorithm": "morphological_pmf",
            "config": {
                "height_threshold": height,
                "bins": bins,
            },
            "ground_level": ground_level,
            "implementation": "pmf_inspired_histogram_baseline",
        },
    )
