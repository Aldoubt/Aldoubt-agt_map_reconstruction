from pathlib import Path

import numpy as np
import open3d as o3d


def load_pcd(path: str) -> np.ndarray:
    """Load PCD file and return Nx3 numpy points."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")

    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point cloud shape: {points.shape}")

    return points
