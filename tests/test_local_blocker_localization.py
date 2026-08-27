import numpy as np

from agt_map_reconstruction.maps.local_blocker_localization import (
    localize_clearance_blocker,
    select_unexpected_failure_targets,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)


def _horizontal_aisle(width_cells=10, length_cells=80, y0=10):
    x0 = 5
    x1 = x0 + length_cells - 1
    y1 = y0 + width_cells - 1
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "centerline_xy": [
            [x0, 0.5 * (y0 + y1)],
            [x1, 0.5 * (y0 + y1)],
        ],
        "width_m": width_cells * 0.05,
        "length_m": length_cells * 0.05,
    }


def test_localizes_interior_hard_clearance_barrier():
    base = np.full((30, 100), UNKNOWN_VALUE, dtype=np.uint8)
    aisle = _horizontal_aisle(width_cells=10)
    base[10:20, 5:85] = FREE_VALUE
    base[10:20, 44:47] = OCCUPIED_VALUE

    result = localize_clearance_blocker(
        base,
        aisle,
        resolution=0.05,
        radius_m=0.20,
    )

    assert result["validation_pass"] is False
    assert result["failure_region"] == "interior"
    assert result["dominant_blocking_source"] == "hard"
    assert result["first_blocker"] is not None
    assert 0.40 <= result["first_blocker"]["start_s_over_l"] <= 0.60
    assert result["longest_blocker"]["length_m"] >= 0.10


def test_localizes_exit_unknown_clearance_barrier():
    base = np.full((30, 100), UNKNOWN_VALUE, dtype=np.uint8)
    aisle = _horizontal_aisle(width_cells=10)
    base[10:20, 5:85] = FREE_VALUE
    base[10:20, 72:75] = UNKNOWN_VALUE

    result = localize_clearance_blocker(
        base,
        aisle,
        resolution=0.05,
        radius_m=0.20,
    )

    assert result["validation_pass"] is False
    assert result["failure_region"] == "exit"
    assert result["dominant_blocking_source"] == "unknown"
    assert result["longest_blocker"]["end_s_over_l"] >= 0.80


def test_selects_first_unexpected_radius_only():
    diagnostics = {
        "aisles": [
            {
                "label": "A01",
                "minimum_clearance_mode": "width_limited",
                "first_unexpected_failed_radius_m": None,
            },
            {
                "label": "A03",
                "minimum_clearance_mode": "connectivity_limited",
                "first_unexpected_failed_radius_m": 0.20,
            },
            {
                "label": "A10",
                "minimum_clearance_mode": "pass",
                "first_unexpected_failed_radius_m": 0.25,
            },
        ]
    }

    assert select_unexpected_failure_targets(diagnostics) == {
        "A03": 0.20,
        "A10": 0.25,
    }
