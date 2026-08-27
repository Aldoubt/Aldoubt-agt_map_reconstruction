import numpy as np

from agt_map_reconstruction.maps.grid_map import build_traversability_map


def test_build_traversability_map_preserves_pcd_grid_origin_and_resolution():
    points = np.array([
        [-1.20, 3.40, 0.10],
        [-1.10, 3.40, 0.20],
        [-1.20, 3.50, 0.15],
    ], dtype=float)

    result = build_traversability_map(points, resolution=0.10, kernel_size=1)

    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata.origin_x == -1.20
    assert metadata.origin_y == 3.40
    assert metadata.resolution == 0.10
    assert metadata.width == result["height"].shape[1]
    assert metadata.height == result["height"].shape[0]


def test_grid_metadata_converts_cell_centres_between_grid_and_world():
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    metadata = GridMetadata(
        resolution=0.10,
        origin_x=-1.20,
        origin_y=3.40,
        width=4,
        height=3,
    )

    assert callable(getattr(metadata, "grid_to_world", None))
    assert callable(getattr(metadata, "world_to_grid", None))

    world = metadata.grid_to_world(2, 1)
    assert np.allclose(world, (-0.95, 3.55))
    assert metadata.world_to_grid(*world) == (2, 1)


def test_grid_metadata_serializes_nav2_compatible_origin_and_shape():
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    metadata = GridMetadata(
        resolution=0.05,
        origin_x=-12.5,
        origin_y=4.25,
        width=797,
        height=912,
        frame_id="map",
        origin_yaw=0.0,
    )

    assert callable(getattr(metadata, "to_dict", None))
    payload = metadata.to_dict()
    assert payload == {
        "frame_id": "map",
        "resolution": 0.05,
        "origin": [-12.5, 4.25, 0.0],
        "width": 797,
        "height": 912,
    }


def test_grid_metadata_applies_origin_yaw_to_coordinate_conversion():
    from math import pi
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    metadata = GridMetadata(
        resolution=1.0,
        origin_x=10.0,
        origin_y=20.0,
        width=3,
        height=3,
        origin_yaw=pi / 2.0,
    )

    world = metadata.grid_to_world(0, 1)
    assert np.allclose(world, (8.5, 20.5))
    assert metadata.world_to_grid(*world) == (0, 1)


def test_grid_metadata_rejects_non_positive_resolution():
    import pytest
    from agt_map_reconstruction.maps.grid_geometry import GridMetadata

    with pytest.raises(ValueError, match="resolution"):
        GridMetadata(
            resolution=0.0,
            origin_x=0.0,
            origin_y=0.0,
            width=1,
            height=1,
        )
