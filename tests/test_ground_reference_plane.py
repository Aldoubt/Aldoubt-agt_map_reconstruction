import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_reference_plane import (
    fit_affine_ground_reference,
)


def _metadata(width=6, height=5):
    return GridMetadata(
        resolution=0.5,
        origin_x=1.0,
        origin_y=-2.0,
        width=width,
        height=height,
        frame_id="map",
    )


def test_affine_ground_reference_recovers_known_plane_with_missing_cells():
    metadata = _metadata()
    yy, xx = np.indices((metadata.height, metadata.width))
    world_x = metadata.origin_x + (xx + 0.5) * metadata.resolution
    world_y = metadata.origin_y + (yy + 0.5) * metadata.resolution
    truth = 0.02 * world_x - 0.01 * world_y + 0.3
    measured = truth.copy()
    measured[:, 3:] = np.nan
    measured[0, 0] = np.nan

    result = fit_affine_ground_reference(measured, metadata)

    np.testing.assert_allclose(result["ground_reference"], truth, atol=1e-6)
    assert result["model"]["support_cell_count"] == int(np.isfinite(measured).sum())
    assert result["model"]["extrapolated_cell_count"] == int(np.isnan(measured).sum())
    assert result["model"]["residual_rmse_m"] < 1e-10
    assert result["model"]["semantic_promotion"] is False


def test_ground_reference_reports_residual_without_semantic_thresholding():
    metadata = _metadata(width=4, height=4)
    yy, xx = np.indices((4, 4))
    surface = 0.1 + 0.01 * xx + 0.02 * yy
    surface[2, 2] += 0.05

    result = fit_affine_ground_reference(surface, metadata)

    assert result["model"]["residual_rmse_m"] > 0.0
    assert result["model"]["residual_p95_abs_m"] > 0.0
    assert result["model"]["semantic_promotion"] is False
