import numpy as np

from agt_map_reconstruction.maps.headland_candidate_geometry import (
    analyze_headland_candidate_geometry,
)


def _row(label, y):
    return {
        "label": label,
        "region_class": "row_aisle",
        "centerline_xy": [[20.0, float(y)], [80.0, float(y)]],
        "polygon_xy": [
            [20.0, float(y) - 2.0],
            [80.0, float(y) - 2.0],
            [80.0, float(y) + 2.0],
            [20.0, float(y) + 2.0],
        ],
    }


def test_cross_row_exit_strip_has_headland_like_geometry():
    regions = [
        _row("A01", 10),
        _row("A02", 20),
        _row("A03", 30),
        {
            "label": "O01",
            "region_class": "wide_open_area_candidate",
            "polygon_xy": [[82.0, 5.0], [90.0, 5.0], [90.0, 35.0], [82.0, 35.0]],
            "centerline_xy": [[86.0, 5.0], [86.0, 35.0]],
        },
    ]

    result = analyze_headland_candidate_geometry(regions, grid_shape=(50, 110))
    item = result["candidates"][0]

    assert item["label"] == "O01"
    assert item["row_axis_alignment"] < 0.2
    assert item["cross_row_overlap_fraction"] > 0.5
    assert item["exit_outward_fraction"] > 0.95
    assert item["entry_outward_fraction"] < 0.05
    assert item["semantic_promotion"] is False


def test_parallel_exterior_band_is_not_cross_row_headland_geometry():
    regions = [
        _row("A01", 10),
        _row("A02", 20),
        _row("A03", 30),
        {
            "label": "O01",
            "region_class": "wide_open_area_candidate",
            "polygon_xy": [[20.0, 40.0], [80.0, 40.0], [80.0, 45.0], [20.0, 45.0]],
            "centerline_xy": [[20.0, 42.5], [80.0, 42.5]],
        },
    ]

    result = analyze_headland_candidate_geometry(regions, grid_shape=(60, 110))
    item = result["candidates"][0]

    assert item["row_axis_alignment"] > 0.95
    assert item["cross_row_overlap_fraction"] < 0.05
    assert item["entry_outward_fraction"] < 0.05
    assert item["exit_outward_fraction"] < 0.05


def test_row_axis_is_consistently_oriented_from_row_centerlines():
    regions = [_row("A01", 10), _row("A02", 20), _row("A03", 30)]
    result = analyze_headland_candidate_geometry(regions, grid_shape=(50, 110))

    assert np.allclose(result["row_axis_direction"], [1.0, 0.0])
    assert np.allclose(result["cross_row_direction"], [0.0, 1.0])
    assert result["row_aisle_count"] == 3
    assert result["candidate_count"] == 0
