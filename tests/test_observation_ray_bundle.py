import numpy as np
import pytest

from agt_map_reconstruction.maps.observation_ray_bundle import (
    ObservationRayBundle,
    load_observation_ray_bundle,
    validate_observation_ray_bundle,
    write_observation_ray_bundle,
)


def test_observation_ray_bundle_round_trip(tmp_path):
    origins = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    endpoints = np.array([[2.0, 0.0, 0.2], [3.0, 0.0, 0.2]])
    bundle = validate_observation_ray_bundle(
        origins,
        endpoints,
        frame_id="map",
        timestamp_s=[1.0, 1.1],
        scan_index=[0, 1],
    )
    path = tmp_path / "observation_rays.npz"
    write_observation_ray_bundle(path, bundle)

    loaded = load_observation_ray_bundle(path)
    assert loaded.ray_count == 2
    assert loaded.frame_id == "map"
    np.testing.assert_allclose(loaded.ray_origin_xyz_m, origins)
    np.testing.assert_allclose(loaded.ray_endpoint_xyz_m, endpoints)
    np.testing.assert_allclose(loaded.timestamp_s, [1.0, 1.1])
    np.testing.assert_array_equal(loaded.scan_index, [0, 1])


def test_rejects_mismatched_ray_arrays():
    with pytest.raises(ValueError, match="matching shapes"):
        validate_observation_ray_bundle(
            np.zeros((2, 3)),
            np.zeros((3, 3)),
        )


def test_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_observation_ray_bundle(
            np.zeros((2, 3)),
            np.ones((2, 3)),
            timestamp_s=[2.0, 1.0],
        )
