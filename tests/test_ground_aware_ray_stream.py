import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_aware_ray_evidence import GroundAwareRayConfig
from agt_map_reconstruction.maps.ground_aware_ray_stream import (
    accumulate_ground_aware_ray_batches,
)
from agt_map_reconstruction.maps.observation_ray_bundle import (
    validate_observation_ray_bundle,
)


def _metadata():
    return GridMetadata(
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        width=8,
        height=3,
        frame_id="map",
    )


def _one_ray():
    return validate_observation_ray_bundle(
        [[0.5, 1.5, 0.20]],
        [[5.5, 1.5, 0.20]],
    )


def test_support_threshold_is_applied_after_cross_chunk_accumulation():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    config = GroundAwareRayConfig(
        min_ground_relative_height_m=0.10,
        max_ground_relative_height_m=0.40,
        min_support_rays=2,
    )

    result = accumulate_ground_aware_ray_batches(
        [_one_ray(), _one_ray()],
        ground,
        metadata,
        config,
    )

    assert result["support_count"][1, 3] == 2
    assert result["support_mask"][1, 3]
    assert result["summary"]["batch_count"] == 2
    assert result["summary"]["input_ray_count"] == 2
    assert result["summary"]["min_support_rays_applied_after_global_accumulation"] is True


def test_streamed_count_matches_expected_hit_cell_policy():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    config = GroundAwareRayConfig(
        min_ground_relative_height_m=0.10,
        max_ground_relative_height_m=0.40,
        min_support_rays=1,
    )

    result = accumulate_ground_aware_ray_batches(
        [_one_ray(), _one_ray()],
        ground,
        metadata,
        config,
    )

    np.testing.assert_array_equal(
        result["support_count"][1, :6],
        [2, 2, 2, 2, 2, 0],
    )
    assert result["summary"]["supported_cell_count"] == 5
