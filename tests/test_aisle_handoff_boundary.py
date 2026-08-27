import numpy as np

from agt_map_reconstruction.maps.aisle_handoff_boundary import (
    estimate_aisle_handoff_boundary,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)


def _horizontal_aisle(width_m=1.0):
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[10.0, 20.0], [110.0, 20.0], [110.0, 30.0], [10.0, 30.0]],
        "centerline_xy": [[10.0, 25.0], [110.0, 25.0]],
        "centerline_map_xy_m": [[1.0, 2.0], [11.0, 2.0]],
        "width_m": float(width_m),
        "length_m": 10.0,
        "heading_rad": 0.0,
    }


def test_exit_unknown_barrier_moves_handoff_before_raw_aisle_end():
    base = np.full((60, 130), FREE_VALUE, dtype=np.uint8)
    base[15:36, 84:89] = UNKNOWN_VALUE

    result = estimate_aisle_handoff_boundary(
        base,
        _horizontal_aisle(),
        resolution=0.10,
        radius_m=0.20,
    )

    assert result["status"] == "ok"
    assert result["component_selection"] == "midpoint"
    assert result["width_clearance_eligible"] is True
    assert np.isclose(result["required_diameter_m"], 0.40)
    assert result["entry_handoff"]["s_over_l"] < 0.05
    assert 0.65 < result["exit_handoff"]["s_over_l"] < 0.75
    assert result["exit_handoff"]["boundary_nearest_source"] == "unknown"
    assert result["exit_transition_length_m"] > 2.5
    assert result["exit_transition"]["dominant_source"] == "unknown"
    assert result["exit_transition"]["direct_unknown_cell_count"] > 0
    assert result["exit_transition"]["direct_hard_cell_count"] == 0
    assert 0.60 < result["row_core_fraction"] < 0.80
    assert result["exit_handoff"]["clearance_m"] >= 0.20


def test_entry_hard_barrier_moves_entry_handoff_after_raw_aisle_start():
    base = np.full((60, 130), FREE_VALUE, dtype=np.uint8)
    base[15:36, 30:35] = OCCUPIED_VALUE

    result = estimate_aisle_handoff_boundary(
        base,
        _horizontal_aisle(),
        resolution=0.10,
        radius_m=0.20,
    )

    assert result["status"] == "ok"
    assert 0.25 < result["entry_handoff"]["s_over_l"] < 0.35
    assert result["entry_handoff"]["boundary_nearest_source"] == "hard"
    assert result["entry_transition_length_m"] > 2.5
    assert result["entry_transition"]["dominant_source"] == "hard"
    assert result["entry_transition"]["direct_hard_cell_count"] > 0
    assert result["entry_transition"]["direct_unknown_cell_count"] == 0
    assert result["entry_handoff"]["clearance_m"] >= 0.20
    assert result["exit_handoff"]["s_over_l"] > 0.95


def test_exit_handoff_uses_safe_lateral_pose_when_centerline_is_blocked():
    base = np.full((60, 130), FREE_VALUE, dtype=np.uint8)
    base[23:28, 106:111] = UNKNOWN_VALUE

    result = estimate_aisle_handoff_boundary(
        base,
        _horizontal_aisle(),
        resolution=0.10,
        radius_m=0.20,
    )

    assert result["status"] == "ok"
    assert result["exit_handoff"]["s_over_l"] > 0.95
    assert result["exit_handoff"]["clearance_m"] >= 0.20
    assert abs(result["exit_handoff"]["cross_track_offset_m"]) >= 0.10
    assert result["exit_handoff"]["boundary_nearest_source"] == "unknown"
    x, y = np.rint(result["exit_handoff"]["grid_xy"]).astype(int)
    assert base[y, x] == FREE_VALUE


def test_existing_safe_fragment_is_not_confused_with_width_eligibility():
    base = np.full((60, 130), UNKNOWN_VALUE, dtype=np.uint8)
    base[20:31, 10:111] = FREE_VALUE

    result = estimate_aisle_handoff_boundary(
        base,
        _horizontal_aisle(width_m=0.35),
        resolution=0.10,
        radius_m=0.20,
    )

    assert result["status"] == "ok"
    assert result["width_clearance_eligible"] is False
    assert np.isclose(result["aisle_width_m"], 0.35)
    assert np.isclose(result["required_diameter_m"], 0.40)
    assert 0.0 < result["row_core_fraction"] <= 1.0


def test_reports_no_safe_component_when_clearance_exceeds_aisle_width():
    base = np.full((60, 130), UNKNOWN_VALUE, dtype=np.uint8)
    base[20:31, 10:111] = FREE_VALUE

    result = estimate_aisle_handoff_boundary(
        base,
        _horizontal_aisle(),
        resolution=0.10,
        radius_m=0.65,
    )

    assert result["status"] == "no_safe_component"
    assert result["width_clearance_eligible"] is False
    assert np.isclose(result["required_diameter_m"], 1.30)
    assert result["row_core_fraction"] == 0.0
    assert result["entry_handoff"] is None
    assert result["exit_handoff"] is None
    assert result["entry_transition"] is None
    assert result["exit_transition"] is None
