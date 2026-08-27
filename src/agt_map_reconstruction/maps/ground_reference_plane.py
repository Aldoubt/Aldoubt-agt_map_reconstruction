"""Fit an explicit affine ground reference from confirmed finite ground support.

This model is geometry-only. Its extrapolated values are never semantic free-space
evidence by themselves; they only provide a local height reference for later 3D
ray observations.
"""

from __future__ import annotations

import numpy as np


def _world_xy_from_cells(xx, yy, metadata):
    local_x = (xx.astype(np.float64) + 0.5) * float(metadata.resolution)
    local_y = (yy.astype(np.float64) + 0.5) * float(metadata.resolution)
    c = float(np.cos(metadata.origin_yaw))
    s = float(np.sin(metadata.origin_yaw))
    world_x = float(metadata.origin_x) + c * local_x - s * local_y
    world_y = float(metadata.origin_y) + s * local_x + c * local_y
    return world_x, world_y


def fit_affine_ground_reference(ground_surface, metadata):
    """Fit z = ax + by + c to finite ground support and extrapolate to the grid.

    No residual threshold is used for semantic promotion. Residual statistics are
    returned so the caller can decide whether an affine model is adequate for the
    dataset before using it as a ray-height reference.
    """
    ground = np.asarray(ground_surface, dtype=np.float64)
    expected_shape = (int(metadata.height), int(metadata.width))
    if ground.shape != expected_shape:
        raise ValueError(
            f"ground_surface shape {ground.shape} does not match grid {expected_shape}"
        )

    finite = np.isfinite(ground)
    yy, xx = np.nonzero(finite)
    if len(xx) < 3:
        raise ValueError("at least three finite ground cells are required")

    world_x, world_y = _world_xy_from_cells(xx, yy, metadata)
    design = np.column_stack((world_x, world_y, np.ones(len(xx), dtype=float)))
    z = ground[yy, xx]
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if int(rank) < 3:
        raise ValueError("finite ground support is degenerate for affine plane fitting")

    predicted_support = design @ coefficients
    residual = z - predicted_support
    abs_residual = np.abs(residual)

    full_y, full_x = np.indices(expected_shape)
    full_world_x, full_world_y = _world_xy_from_cells(
        full_x.reshape(-1),
        full_y.reshape(-1),
        metadata,
    )
    reference = (
        coefficients[0] * full_world_x
        + coefficients[1] * full_world_y
        + coefficients[2]
    ).reshape(expected_shape)

    return {
        "ground_reference": reference.astype(np.float32),
        "finite_support_mask": finite,
        "model": {
            "schema_version": 1,
            "model_type": "affine_plane",
            "equation": "z_m = a*x_m + b*y_m + c",
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c": float(coefficients[2]),
            "support_cell_count": int(len(xx)),
            "grid_cell_count": int(ground.size),
            "extrapolated_cell_count": int(ground.size - len(xx)),
            "residual_rmse_m": float(np.sqrt(np.mean(residual * residual))),
            "residual_median_abs_m": float(np.median(abs_residual)),
            "residual_p95_abs_m": float(np.percentile(abs_residual, 95.0)),
            "residual_max_abs_m": float(np.max(abs_residual)),
            "semantic_promotion": False,
        },
    }
