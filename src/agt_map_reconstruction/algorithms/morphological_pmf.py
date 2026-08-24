"""Lightweight PMF-inspired baseline.

This is intentionally dependency-free. It provides a comparable morphology
baseline before integrating a full PMF implementation.
"""

import numpy as np


def segment(points: np.ndarray, config=None):
    config = config or {}
    height = float(config.get("height_threshold", 0.25))
    z = points[:, 2]
    bins = int(config.get("bins", 200))
    hist, edges = np.histogram(z, bins=bins)
    ground_level = edges[np.argmax(hist)]
    mask = z < ground_level + height
    return {
        "ground": points[mask],
        "non_ground": points[~mask],
        "name": "morphological_pmf",
    }
