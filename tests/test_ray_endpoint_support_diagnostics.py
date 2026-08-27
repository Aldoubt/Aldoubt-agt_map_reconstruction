import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, UNKNOWN_VALUE
from agt_map_reconstruction.maps.ray_endpoint_support_diagnostics import (
    analyze_endpoint_support_threshold,
    sweep_endpoint_support_thresholds,
)


def _baseline():
    return {
        "radius_m": 1.0,
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "row_cross_span": [1.0, 4.0],
        "sides": {
            "entry": {
                "endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 3.0}
            },
            "exit": {
                "endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 6.0}
            },
        },
    }


def test_endpoint_support_is_localized_without_touching_base_map():
    base = np.full((6, 10), UNKNOWN_VALUE, dtype=np.uint8)
    base[:, 3:7] = FREE_VALUE
    frozen = base.copy()
    count = np.zeros_like(base, dtype=np.uint32)
    count[2, 2] = 3
    count[3, 2] = 3
    count[2, 7] = 1

    result = analyze_endpoint_support_threshold(
        base,
        count,
        _baseline(),
        min_support_rays=1,
        resolution=1.0,
    )

    assert np.array_equal(base, frozen)
    assert result["supported_unknown_cell_count"] == 3
    assert result["sides"]["entry"]["supported_unknown_cell_count"] == 2
    assert result["sides"]["exit"]["supported_unknown_cell_count"] == 1
    assert result["semantic_promotion"] is False


def test_support_threshold_sweep_reduces_sparse_unknown_support():
    base = np.full((6, 10), UNKNOWN_VALUE, dtype=np.uint8)
    base[:, 3:7] = FREE_VALUE
    count = np.zeros_like(base, dtype=np.uint32)
    count[2, 2] = 3
    count[3, 2] = 2
    count[2, 7] = 1

    result = sweep_endpoint_support_thresholds(
        base,
        count,
        _baseline(),
        min_support_values=[1, 2, 3],
        resolution=1.0,
    )

    assert [item["supported_unknown_cell_count"] for item in result["thresholds"]] == [3, 2, 1]
    assert result["automatic_threshold_selection"] is False
    assert result["semantic_promotion"] is False
