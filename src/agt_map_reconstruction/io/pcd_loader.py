from pathlib import Path
import open3d as o3d


def load_pcd(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    return cloud
