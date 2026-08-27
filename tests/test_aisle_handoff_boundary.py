import numpy as np

from agt_map_reconstruction.maps.aisle_handoff_boundary import (
    estimate_aisle_handoff_boundary,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)


def _horizontal_aisle():
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[10.0, 20.0], [110.0, 20.0], [110.0, 30.0], [10.0, 30.0]],
        "centerline_xy": [[10.0, 25.0], [110.0, 25.0]],
        "centerline_map_xy_m": [[1.0, 2.0], [11.0, 2.0]],
        "width_m": 1.0,
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
    assert result["entry_handoff"]["s_over_l"] < 0.05
    assert 0.65 < result["exit_handoff"]["s_over_l"] < 0.75
    assert result["exit_handoff"]["boundary_source"] == "unknown"
    assert result["exit_transition_length_m"] > 2.5
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
    assert result["entry_handoff"]["boundary_source"] == "hard"
    assert result["entry_transition_length_m"] > 2.5
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
    x, y = np.rint(result["exit_handoff"]["grid_xy"]).astype(int)
    assert base[y, x] == FREE_VALUE


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
    assert result["entry_handoff"] is None
    assert result["exit_handoff"] is None
