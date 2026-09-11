"""PCD output helpers."""

from pathlib import Path

import numpy as np
import open3d as o3d


def write_pcd(path: str | Path, points: np.ndarray, *, compressed: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={points.shape}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    ok = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=not compressed,
        compressed=compressed,
    )
    if not ok:
        raise IOError(f"Failed to write point cloud: {path}")
    return path
