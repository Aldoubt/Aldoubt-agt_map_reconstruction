from dataclasses import dataclass
import numpy as np


@dataclass
class SegmentationResult:
    ground: np.ndarray
    non_ground: np.ndarray


def height_threshold(points: np.ndarray, threshold: float = 0.15):
    """Simple agricultural baseline.

    Assumes normalized ground is close to the minimum local elevation.
    """
    z = points[:, 2]
    ground_level = np.percentile(z, 10)
    mask = np.abs(z - ground_level) < threshold
    return SegmentationResult(points[mask], points[~mask])


def voxel_ground(points: np.ndarray, voxel_size: float = 0.2):
    """Placeholder interface for voxel based ground methods.

    Advanced methods (PMF/CSF/Patchwork) will share this interface.
    """
    return height_threshold(points)
