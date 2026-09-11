"""Canonical contracts for ground-segmentation plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def validate_points(points: np.ndarray) -> np.ndarray:
    """Return a validated Nx3 floating-point view of a point cloud."""
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("points must contain numeric values")
    return array


@dataclass(frozen=True)
class SegmentationResult:
    """Stable output contract shared by every segmentation plugin.

    The ground/non_ground properties and mapping-style accessors intentionally
    preserve compatibility with the repository's older visualization helpers.
    """

    ground_points: np.ndarray
    non_ground_points: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ground = validate_points(self.ground_points)
        non_ground = validate_points(self.non_ground_points)
        object.__setattr__(self, "ground_points", ground)
        object.__setattr__(self, "non_ground_points", non_ground)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ground(self) -> np.ndarray:
        return self.ground_points

    @property
    def non_ground(self) -> np.ndarray:
        return self.non_ground_points

    def __getitem__(self, key: str):
        aliases = {
            "ground": self.ground_points,
            "ground_points": self.ground_points,
            "non_ground": self.non_ground_points,
            "non_ground_points": self.non_ground_points,
            "metadata": self.metadata,
            "name": self.metadata.get("algorithm"),
        }
        if key not in aliases:
            raise KeyError(key)
        return aliases[key]


AlgorithmConfig = Mapping[str, Any]
