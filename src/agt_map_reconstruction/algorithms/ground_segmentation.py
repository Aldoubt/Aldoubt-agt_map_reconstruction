"""Backward-compatible wrappers for the pre-registry API.

New code should use agt_map_reconstruction.algorithms.run_algorithm().
"""

from __future__ import annotations

import numpy as np

from .base import SegmentationResult
from .registry import run_algorithm


def height_threshold(points: np.ndarray, threshold: float = 0.15) -> SegmentationResult:
    return run_algorithm(
        "height_threshold",
        points,
        {"height_threshold": float(threshold)},
    )


def voxel_ground(points: np.ndarray, voxel_size: float = 0.2) -> SegmentationResult:
    """Legacy placeholder retained for callers of the original API.

    voxel_size is recorded for compatibility but no voxel segmentation is
    performed. The function delegates to the height-threshold baseline.
    """
    result = run_algorithm("height_threshold", points)
    metadata = dict(result.metadata)
    metadata["legacy_wrapper"] = "voxel_ground"
    metadata["voxel_size"] = float(voxel_size)
    return SegmentationResult(result.ground_points, result.non_ground_points, metadata)
