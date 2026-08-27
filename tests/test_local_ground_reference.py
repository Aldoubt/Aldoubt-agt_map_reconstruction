import numpy as np
import pytest

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.local_ground_reference import (
    fit_knn_local_affine_ground_reference,
)


def _metadata(width, height, resolution=0.10):
    return GridMetadata(
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        width=width,
        height=height,
        frame_id="map",
    )


def test_local_affine_reference_tracks_smooth_nonplanar_ground_better_than_global_plane():
    height = 36
    width = 48
    metadata = _metadata(width, height)
    yy, xx = np.indices((height, width), dtype=float)
    x_m = (xx + 0.5) * metadata.resolution
    y_m = (yy + 0.5) * metadata.resolution
    ground = (
        0.08 * np.sin(0.8 * x_m)
        + 0.05 * np.cos(0.6 * y_m)
        + 0.01 * x_m
    )

    result = fit_knn_local_affine_ground_reference(
        ground,
        metadata,
        neighbor_count=16,
        chunk_size=256,
    )

    assert result["model"]["cv_residual_rmse_m"] < result["global_affine_baseline"]["residual_rmse_m"]
    assert result["model"]["cv_residual_p95_abs_m"] < result["global_affine_baseline"]["residual_p95_abs_m"]
    assert result["model"]["invalid_fit_cell_count"] == 0


def test_unknown_cells_receive_geometry_reference_and_distance_to_observed_support():
    metadata = _metadata(20, 16)
    yy, xx = np.indices((16, 20), dtype=float)
    ground = 0.02 * xx + 0.01 * yy
    ground[:, 8:12] = np.nan

    result = fit_knn_local_affine_ground_reference(
        ground,
        metadata,
        neighbor_count=12,
        chunk_size=128,
    )

    reference = result["ground_reference"]
    distance = result["nearest_support_distance_m"]
    assert np.all(np.isfinite(reference[:, 8:12]))
    assert np.all(distance[:, 8:12] > 0.0)
    assert result["model"]["unknown_cell_count"] == 16 * 4
    assert result["model"]["unknown_nearest_support_distance_p95_m"] > 0.0
    assert result["model"]["semantic_promotion"] is False


def test_neighbor_count_requires_enough_local_support_for_a_plane():
    metadata = _metadata(8, 8)
    ground = np.zeros((8, 8), dtype=float)

    with pytest.raises(ValueError, match="neighbor_count"):
        fit_knn_local_affine_ground_reference(
            ground,
            metadata,
            neighbor_count=2,
        )
