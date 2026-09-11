"""Deterministic offline point-cloud cleaning.

This module absorbs the useful intent of the historical
`agt_pointcloud_map_tools` repository without carrying its ROS 2 runtime
skeleton into the offline map-reconstruction stack.

The pipeline is intentionally conservative:

1. voxel deduplication/downsampling;
2. optional radius-outlier removal;
3. optional Euclidean/DBSCAN component-size filtering.

A single fused PCD does not contain enough temporal evidence to reliably
classify a sparse cluster as a dynamic human/vehicle ghost. Dynamic-artifact
removal therefore remains an explicit upstream/temporal stage rather than a
misleading switch in this module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RadiusOutlierConfig:
    enabled: bool = True
    radius: float = 0.30
    min_neighbors: int = 8


@dataclass(frozen=True)
class ClusterFilterConfig:
    enabled: bool = False
    tolerance: float = 0.25
    min_cluster_size: int = 20
    max_cluster_size: int | None = None


@dataclass(frozen=True)
class CleaningConfig:
    voxel_size: float = 0.10
    radius_outlier: RadiusOutlierConfig = field(default_factory=RadiusOutlierConfig)
    cluster_filter: ClusterFilterConfig = field(default_factory=ClusterFilterConfig)


@dataclass
class CleaningResult:
    points: np.ndarray
    removed_points: np.ndarray
    metadata: dict[str, Any]


def _as_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points.shape}")
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points = points[finite]
    return points


def _split_by_mask(points: np.ndarray, keep_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points[keep_mask], points[~keep_mask]


def voxel_filter(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first input point in every voxel and return removed duplicates."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if len(points) == 0:
        return points.copy(), points.copy()

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    keep_mask = np.zeros(len(points), dtype=bool)
    keep_mask[np.sort(first_indices)] = True
    return _split_by_mask(points, keep_mask)


def radius_outlier_filter(
    points: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove points with too few neighbors inside radius.

    The point itself is included in the neighbor count, matching the simple
    historical configuration semantics.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")
    if len(points) == 0:
        return points.copy(), points.copy()

    tree = cKDTree(points)
    try:
        counts = tree.query_ball_point(points, radius, return_length=True)
    except TypeError:
        counts = np.fromiter(
            (len(indices) for indices in tree.query_ball_point(points, radius)),
            dtype=np.int64,
            count=len(points),
        )
    counts = np.asarray(counts)
    return _split_by_mask(points, counts >= min_neighbors)


def cluster_size_filter(
    points: np.ndarray,
    tolerance: float,
    min_cluster_size: int,
    max_cluster_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter connected Euclidean components by point count using Open3D DBSCAN."""
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if max_cluster_size is not None and max_cluster_size < min_cluster_size:
        raise ValueError("max_cluster_size must be >= min_cluster_size")
    if len(points) == 0:
        return points.copy(), points.copy()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    labels = np.asarray(
        cloud.cluster_dbscan(eps=tolerance, min_points=1, print_progress=False),
        dtype=np.int64,
    )
    if len(labels) != len(points):
        raise RuntimeError("Open3D returned an unexpected cluster label count")

    _, counts = np.unique(labels, return_counts=True)
    sizes = counts[labels]
    keep_mask = sizes >= min_cluster_size
    if max_cluster_size is not None:
        keep_mask &= sizes <= max_cluster_size
    return _split_by_mask(points, keep_mask)


def clean_points(points: np.ndarray, config: CleaningConfig) -> CleaningResult:
    points = _as_points(points)
    original_count = len(points)
    removed_batches: list[np.ndarray] = []
    stages: list[dict[str, Any]] = []

    current, removed = voxel_filter(points, config.voxel_size)
    removed_batches.append(removed)
    stages.append(
        {
            "stage": "voxel",
            "input_points": original_count,
            "output_points": len(current),
            "removed_points": len(removed),
            "voxel_size": config.voxel_size,
        }
    )

    if config.radius_outlier.enabled:
        before = len(current)
        current, removed = radius_outlier_filter(
            current,
            config.radius_outlier.radius,
            config.radius_outlier.min_neighbors,
        )
        removed_batches.append(removed)
        stages.append(
            {
                "stage": "radius_outlier",
                "input_points": before,
                "output_points": len(current),
                "removed_points": len(removed),
                "radius": config.radius_outlier.radius,
                "min_neighbors": config.radius_outlier.min_neighbors,
            }
        )

    if config.cluster_filter.enabled:
        before = len(current)
        current, removed = cluster_size_filter(
            current,
            config.cluster_filter.tolerance,
            config.cluster_filter.min_cluster_size,
            config.cluster_filter.max_cluster_size,
        )
        removed_batches.append(removed)
        stages.append(
            {
                "stage": "cluster_size",
                "input_points": before,
                "output_points": len(current),
                "removed_points": len(removed),
                "tolerance": config.cluster_filter.tolerance,
                "min_cluster_size": config.cluster_filter.min_cluster_size,
                "max_cluster_size": config.cluster_filter.max_cluster_size,
            }
        )

    non_empty_removed = [batch for batch in removed_batches if len(batch)]
    removed_points = (
        np.concatenate(non_empty_removed, axis=0)
        if non_empty_removed
        else np.empty((0, 3), dtype=np.float32)
    )

    metadata = {
        "input_points": original_count,
        "output_points": len(current),
        "removed_points": len(removed_points),
        "stages": stages,
        "dynamic_ghost_cleaner": {
            "implemented": False,
            "reason": (
                "Single fused PCD geometry is insufficient for reliable dynamic "
                "classification; use temporal/static-evidence experiments instead."
            ),
        },
    }
    return CleaningResult(current, removed_points, metadata)


def load_cleaning_config(path: str | Path) -> CleaningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("preprocessing", raw)

    radius = raw.get("radius_outlier", raw.get("outlier_filter", {})) or {}
    cluster = raw.get("cluster_filter", {}) or {}

    max_cluster_size = cluster.get("max_cluster_size")
    if max_cluster_size in ("", 0):
        max_cluster_size = None

    return CleaningConfig(
        voxel_size=float(raw.get("voxel_size", 0.10)),
        radius_outlier=RadiusOutlierConfig(
            enabled=bool(radius.get("enabled", True)),
            radius=float(radius.get("radius", 0.30)),
            min_neighbors=int(radius.get("min_neighbors", 8)),
        ),
        cluster_filter=ClusterFilterConfig(
            enabled=bool(cluster.get("enabled", False)),
            tolerance=float(cluster.get("tolerance", 0.25)),
            min_cluster_size=int(cluster.get("min_cluster_size", 20)),
            max_cluster_size=(
                int(max_cluster_size) if max_cluster_size is not None else None
            ),
        ),
    )
