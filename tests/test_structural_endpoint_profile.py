import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.structural_endpoint_profile import (
    build_structural_support_profile,
)


def _aisle(reverse=False):
    line = [[5.0, 30.0], [70.0, 30.0]]
    if reverse:
        line = list(reversed(line))
    return {
        "label": "A01",
        "polygon_xy": [
            [5.0, 25.0],
            [70.0, 25.0],
            [70.0, 35.0],
            [5.0, 35.0],
        ],
        "centerline_xy": line,
    }


def _bilateral_structure_map():
    base = np.full((60, 80), FREE_VALUE, dtype=np.uint8)
    # Bilateral ridge/plant structure exists only over x=[10, 49]. The free
    # aisle itself continues to x=70, so a structural endpoint detector must
    # later be able to stop before the geometric aisle endpoint.
    base[21:25, 10:50] = OCCUPIED_VALUE
    base[36:40, 10:50] = OCCUPIED_VALUE
    return base


def test_profile_tracks_bilateral_hard_support_before_free_aisle_end():
    profile = build_structural_support_profile(
        _bilateral_structure_map(),
        _aisle(),
        resolution_m=0.10,
        strip_width_m=0.40,
        bin_size_m=0.50,
        row_axis=[1.0, 0.0],
    )

    centers = np.asarray(profile["bin_center_grid_xy"], dtype=float)
    left = np.asarray(profile["left_hard_support_fraction"], dtype=float)
    right = np.asarray(profile["right_hard_support_fraction"], dtype=float)

    structural = (centers[:, 0] >= 12.0) & (centers[:, 0] <= 47.0)
    beyond = centers[:, 0] >= 55.0

    assert np.count_nonzero(structural) > 0
    assert np.all(left[structural] > 0.75)
    assert np.all(right[structural] > 0.75)
    assert np.all(left[beyond] == 0.0)
    assert np.all(right[beyond] == 0.0)
    assert profile["policy"]["unknown_counted_as_structural"] is False
    assert profile["policy"]["navigation_map_modified"] is False


def test_unknown_is_reported_separately_and_never_counts_as_structure():
    base = _bilateral_structure_map()
    # Replace right-side structure with UNKNOWN over a longitudinal interval.
    base[36:40, 30:40] = UNKNOWN_VALUE

    profile = build_structural_support_profile(
        base,
        _aisle(),
        resolution_m=0.10,
        strip_width_m=0.40,
        bin_size_m=0.50,
        row_axis=[1.0, 0.0],
    )

    centers = np.asarray(profile["bin_center_grid_xy"], dtype=float)
    right_hard = np.asarray(profile["right_hard_support_fraction"], dtype=float)
    right_unknown = np.asarray(profile["right_unknown_fraction"], dtype=float)
    affected = (centers[:, 0] >= 32.0) & (centers[:, 0] <= 37.0)

    assert np.count_nonzero(affected) > 0
    assert np.all(right_unknown[affected] > 0.75)
    assert np.all(right_hard[affected] == 0.0)


def test_explicit_common_row_axis_normalizes_reversed_centerline_orientation():
    base = _bilateral_structure_map()
    forward = build_structural_support_profile(
        base,
        _aisle(reverse=False),
        resolution_m=0.10,
        strip_width_m=0.40,
        bin_size_m=0.50,
        row_axis=[1.0, 0.0],
    )
    reversed_source = build_structural_support_profile(
        base,
        _aisle(reverse=True),
        resolution_m=0.10,
        strip_width_m=0.40,
        bin_size_m=0.50,
        row_axis=[1.0, 0.0],
    )

    assert np.allclose(forward["row_axis_direction"], [1.0, 0.0])
    assert np.allclose(reversed_source["row_axis_direction"], [1.0, 0.0])
    assert np.allclose(
        forward["left_hard_support_fraction"],
        reversed_source["left_hard_support_fraction"],
    )
    assert np.allclose(
        forward["right_hard_support_fraction"],
        reversed_source["right_hard_support_fraction"],
    )
