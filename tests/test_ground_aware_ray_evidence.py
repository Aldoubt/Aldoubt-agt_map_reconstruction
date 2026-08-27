import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_aware_ray_evidence import (
    GroundAwareRayConfig,
    accumulate_ground_aware_ray_support,
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


def _config(min_support=1):
    return GroundAwareRayConfig(
        min_ground_relative_height_m=0.10,
        max_ground_relative_height_m=0.40,
        min_support_rays=min_support,
    )


def test_low_height_ray_supports_traversed_cells_but_not_hit_cell():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    bundle = validate_observation_ray_bundle(
        [[0.5, 1.5, 0.20]],
        [[5.5, 1.5, 0.20]],
    )

    result = accumulate_ground_aware_ray_support(
        bundle,
        ground,
        metadata,
        _config(),
    )

    # Cells x=0..4 are traversed at low height; x=5 is the return cell.
    np.testing.assert_array_equal(result["support_count"][1, :6], [1, 1, 1, 1, 1, 0])
    assert result["summary"]["supported_cell_count"] == 5


def test_high_canopy_ray_does_not_clear_ground_cells():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    bundle = validate_observation_ray_bundle(
        [[0.5, 1.5, 1.50]],
        [[5.5, 1.5, 1.50]],
    )

    result = accumulate_ground_aware_ray_support(
        bundle,
        ground,
        metadata,
        _config(),
    )

    assert not np.any(result["support_mask"])
    assert result["summary"]["supported_cell_visits"] == 0


def test_missing_ground_reference_cannot_receive_free_support():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    ground[1, 2] = np.nan
    bundle = validate_observation_ray_bundle(
        [[0.5, 1.5, 0.20]],
        [[5.5, 1.5, 0.20]],
    )

    result = accumulate_ground_aware_ray_support(
        bundle,
        ground,
        metadata,
        _config(),
    )

    assert result["support_count"][1, 2] == 0
    assert result["support_count"][1, 1] == 1
    assert result["support_count"][1, 3] == 1


def test_minimum_support_count_is_applied_after_accumulation():
    metadata = _metadata()
    ground = np.zeros((3, 8), dtype=float)
    bundle = validate_observation_ray_bundle(
        [[0.5, 1.5, 0.20], [0.5, 1.5, 0.20]],
        [[5.5, 1.5, 0.20], [5.5, 1.5, 0.20]],
    )

    result = accumulate_ground_aware_ray_support(
        bundle,
        ground,
        metadata,
        _config(min_support=2),
    )

    assert result["support_count"][1, 3] == 2
    assert result["support_mask"][1, 3]
    assert not result["support_mask"][1, 5]
