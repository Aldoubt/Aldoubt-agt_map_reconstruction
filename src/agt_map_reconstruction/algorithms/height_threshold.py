"""Simple height based ground segmentation baseline."""

import numpy as np


def segment(points: np.ndarray, config=None):
    config = config or {}
    threshold = float(config.get("height_threshold", 0.15))
    z0 = np.percentile(points[:, 2], 10)
    ground_mask = points[:, 2] < z0 + threshold
    return {
        "ground": points[ground_mask],
        "non_ground": points[~ground_mask],
        "name": "height_threshold",
    }
