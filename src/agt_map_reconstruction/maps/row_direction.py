"""Estimate agricultural row direction from geometric map structure.

This module provides a geometry-only baseline for agricultural scenes.
The goal is not semantic recognition, but recovering dominant field/crop
orientation before corridor extraction.
"""

import numpy as np


def _principal_direction(points):
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argmax(values)]


def _mean_axial_direction(directions, weights):
    """Average unsigned 2D directions through doubled angles."""
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    vector = np.array([
        np.sum(weights * np.cos(2.0 * angles)),
        np.sum(weights * np.sin(2.0 * angles)),
    ])
    if np.linalg.norm(vector) < 1e-12:
        return directions[np.argmax(weights)]
    angle = 0.5 * np.arctan2(vector[1], vector[0])
    return np.array([np.cos(angle), np.sin(angle)])


def estimate_row_direction(points_or_grid, structure_threshold=None,
                           component_min_cells=20):
    """Estimate dominant 2D direction using PCA.

    Args:
        points_or_grid:
            Either Nx2 points or a 2D grid. For grids, valid cells are
            converted into image coordinates.

    Returns:
        angle_rad: dominant direction angle in radians.
        direction: unit vector [dx, dy].
    """
    data = np.asarray(points_or_grid)

    if data.ndim == 2 and data.shape[1] == 2:
        points = data
    elif data.ndim == 2:
        valid = np.isfinite(data)
        if structure_threshold is not None:
            valid &= data >= structure_threshold
        y, x = np.where(valid)
        points = np.column_stack((x, y))

        if structure_threshold is not None and len(points) >= 2:
            from scipy import ndimage

            labels, count = ndimage.label(
                valid,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            directions = []
            weights = []
            for label, component_slice in enumerate(
                ndimage.find_objects(labels),
                1,
            ):
                if component_slice is None:
                    continue
                component_labels = labels[component_slice]
                component_y, component_x = np.where(component_labels == label)
                if len(component_x) < component_min_cells:
                    continue
                component_y = component_y + component_slice[0].start
                component_x = component_x + component_slice[1].start
                component = np.column_stack((component_x, component_y))
                directions.append(_principal_direction(component))
                weights.append(float(len(component)))
            if directions:
                direction = _mean_axial_direction(
                    np.asarray(directions),
                    np.asarray(weights),
                )
                angle = float(np.arctan2(direction[1], direction[0]))
                return angle, direction
    else:
        raise ValueError("input must be Nx2 points or 2D grid")

    if len(points) < 2:
        return 0.0, np.array([1.0, 0.0])

    direction = _principal_direction(points)

    angle = float(np.arctan2(direction[1], direction[0]))
    return angle, direction
