"""Local affine ground-height reference for trajectory-aware ray evidence.

The model is geometry-only. It extrapolates a height reference from nearby finite
PCD-derived ground support, but never promotes a cell to semantic free space by
itself. Confidence is reported through leave-one-out residuals and distance to
nearest observed ground support.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .ground_reference_plane import fit_affine_ground_reference


def _world_xy_from_cells(xx, yy, metadata):
    local_x = (np.asarray(xx, dtype=np.float64) + 0.5) * float(metadata.resolution)
    local_y = (np.asarray(yy, dtype=np.float64) + 0.5) * float(metadata.resolution)
    c = float(np.cos(metadata.origin_yaw))
    s = float(np.sin(metadata.origin_yaw))
    world_x = float(metadata.origin_x) + c * local_x - s * local_y
    world_y = float(metadata.origin_y) + s * local_x + c * local_y
    return np.column_stack((world_x.reshape(-1), world_y.reshape(-1)))


def _predict_local_affine(
    tree,
    support_xy,
    support_z,
    target_xy,
    neighbor_count,
    chunk_size,
    exclude_self=False,
):
    target_xy = np.asarray(target_xy, dtype=np.float64)
    prediction = np.full(target_xy.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(target_xy.shape[0], dtype=bool)
    nearest_distance = np.full(target_xy.shape[0], np.nan, dtype=np.float64)
    query_count = int(neighbor_count) + (1 if exclude_self else 0)

    for start in range(0, target_xy.shape[0], int(chunk_size)):
        end = min(target_xy.shape[0], start + int(chunk_size))
        targets = target_xy[start:end]
        distances, indices = tree.query(targets, k=query_count)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        if exclude_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        nearest_distance[start:end] = distances[:, 0]
        neighbor_xy = support_xy[indices]
        neighbor_z = support_z[indices]
        delta = neighbor_xy - targets[:, None, :]
        x = delta[:, :, 0]
        y = delta[:, :, 1]

        normal = np.empty((targets.shape[0], 3, 3), dtype=np.float64)
        normal[:, 0, 0] = np.sum(x * x, axis=1)
        normal[:, 0, 1] = normal[:, 1, 0] = np.sum(x * y, axis=1)
        normal[:, 0, 2] = normal[:, 2, 0] = np.sum(x, axis=1)
        normal[:, 1, 1] = np.sum(y * y, axis=1)
        normal[:, 1, 2] = normal[:, 2, 1] = np.sum(y, axis=1)
        normal[:, 2, 2] = float(neighbor_count)

        rhs = np.column_stack((
            np.sum(x * neighbor_z, axis=1),
            np.sum(y * neighbor_z, axis=1),
            np.sum(neighbor_z, axis=1),
        ))

        rank = np.linalg.matrix_rank(normal)
        local_valid = rank == 3
        if np.any(local_valid):
            # numpy.solve treats a stacked RHS of shape (B, 3) ambiguously as
            # matrix-valued input. Keep an explicit singleton column and squeeze
            # it after the batched solve.
            coeff = np.linalg.solve(
                normal[local_valid],
                rhs[local_valid, :, None],
            )[:, :, 0]
            local_prediction = np.full(targets.shape[0], np.nan, dtype=np.float64)
            # The coordinate system is centred at each target, so the intercept
            # is the predicted height at that target.
            local_prediction[local_valid] = coeff[:, 2]
            prediction[start:end] = local_prediction
        valid[start:end] = local_valid

    return prediction, nearest_distance, valid


def _residual_summary(residual):
    residual = np.asarray(residual, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {
            "rmse_m": None,
            "median_abs_m": None,
            "p95_abs_m": None,
            "max_abs_m": None,
            "sample_count": 0,
        }
    abs_residual = np.abs(residual)
    return {
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "median_abs_m": float(np.median(abs_residual)),
        "p95_abs_m": float(np.percentile(abs_residual, 95.0)),
        "max_abs_m": float(np.max(abs_residual)),
        "sample_count": int(residual.size),
    }


def fit_knn_local_affine_ground_reference(
    ground_surface,
    metadata,
    neighbor_count,
    chunk_size=20_000,
):
    """Fit a KNN local affine height reference over the whole map grid.

    `neighbor_count` is intentionally explicit rather than paper-frozen. Model
    quality is evaluated by leave-one-out prediction on all finite ground cells.
    The returned nearest-support-distance grid must be carried into later ray
    evidence so extrapolation distance can remain explicit.
    """
    ground = np.asarray(ground_surface, dtype=np.float64)
    expected_shape = (int(metadata.height), int(metadata.width))
    if ground.shape != expected_shape:
        raise ValueError(
            f"ground_surface shape {ground.shape} does not match grid {expected_shape}"
        )
    if not isinstance(neighbor_count, (int, np.integer)) or int(neighbor_count) < 3:
        raise ValueError("neighbor_count must be an integer >= 3")
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")

    finite = np.isfinite(ground)
    support_y, support_x = np.nonzero(finite)
    support_count = int(len(support_x))
    if support_count <= int(neighbor_count):
        raise ValueError("finite support must exceed neighbor_count for leave-one-out validation")

    support_xy = _world_xy_from_cells(support_x, support_y, metadata)
    support_z = ground[support_y, support_x]
    tree = cKDTree(support_xy)

    full_y, full_x = np.indices(expected_shape)
    full_xy = _world_xy_from_cells(full_x.reshape(-1), full_y.reshape(-1), metadata)
    reference_flat, nearest_flat, valid_flat = _predict_local_affine(
        tree,
        support_xy,
        support_z,
        full_xy,
        int(neighbor_count),
        int(chunk_size),
        exclude_self=False,
    )

    cv_prediction, _, cv_valid = _predict_local_affine(
        tree,
        support_xy,
        support_z,
        support_xy,
        int(neighbor_count),
        int(chunk_size),
        exclude_self=True,
    )
    cv_residual = np.where(cv_valid, support_z - cv_prediction, np.nan)
    cv = _residual_summary(cv_residual)

    reference = reference_flat.reshape(expected_shape).astype(np.float32)
    nearest = nearest_flat.reshape(expected_shape).astype(np.float32)
    valid_mask = valid_flat.reshape(expected_shape)
    unknown = ~finite
    unknown_distances = nearest[unknown & np.isfinite(nearest)]

    global_model = fit_affine_ground_reference(ground, metadata)["model"]
    model = {
        "schema_version": 1,
        "model_type": "knn_local_affine",
        "neighbor_count": int(neighbor_count),
        "support_cell_count": support_count,
        "unknown_cell_count": int(np.count_nonzero(unknown)),
        "invalid_fit_cell_count": int(np.count_nonzero(~valid_mask)),
        "cv_valid_support_cell_count": int(np.count_nonzero(cv_valid)),
        "cv_residual_rmse_m": cv["rmse_m"],
        "cv_residual_median_abs_m": cv["median_abs_m"],
        "cv_residual_p95_abs_m": cv["p95_abs_m"],
        "cv_residual_max_abs_m": cv["max_abs_m"],
        "unknown_nearest_support_distance_median_m": (
            None if unknown_distances.size == 0 else float(np.median(unknown_distances))
        ),
        "unknown_nearest_support_distance_p95_m": (
            None if unknown_distances.size == 0 else float(np.percentile(unknown_distances, 95.0))
        ),
        "unknown_nearest_support_distance_max_m": (
            None if unknown_distances.size == 0 else float(np.max(unknown_distances))
        ),
        "semantic_promotion": False,
    }
    return {
        "ground_reference": reference,
        "nearest_support_distance_m": nearest,
        "valid_fit_mask": valid_mask,
        "finite_support_mask": finite,
        "model": model,
        "global_affine_baseline": global_model,
    }
