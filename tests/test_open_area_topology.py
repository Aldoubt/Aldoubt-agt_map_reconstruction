import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.open_area_topology import (
    analyze_handoff_open_area_topology,
)


def _handoff(label="A01", xy=(10.0, 15.0), radius=0.10):
    return {
        "label": label,
        "radius_m": radius,
        "status": "ok",
        "width_clearance_eligible": True,
        "entry_handoff": {"grid_xy": [5.0, 15.0]},
        "exit_handoff": {"grid_xy": [float(xy[0]), float(xy[1])]},
    }


def _open_region():
    return {
        "label": "O01",
        "region_class": "wide_open_area_candidate",
        "polygon_xy": [[30.0, 10.0], [45.0, 10.0], [45.0, 20.0], [30.0, 20.0]],
    }


def _exit_result(result):
    return next(item for item in result["handoffs"] if item["side"] == "exit")


def test_reports_strict_connection_to_open_area():
    base = np.full((30, 60), FREE_VALUE, dtype=np.uint8)

    result = analyze_handoff_open_area_topology(
        base,
        [_handoff()],
        [_open_region()],
        resolution=0.10,
        radius_m=0.10,
    )

    exit_result = _exit_result(result)
    assert exit_result["connectivity_class"] == "strict_connected"
    assert exit_result["strict_connected_candidates"] == ["O01"]
    assert exit_result["unknown_bridge_candidates"] == []


def test_reports_unknown_bridge_only_when_unknown_strip_is_needed():
    base = np.full((30, 60), FREE_VALUE, dtype=np.uint8)
    base[:, 20:25] = UNKNOWN_VALUE

    result = analyze_handoff_open_area_topology(
        base,
        [_handoff()],
        [_open_region()],
        resolution=0.10,
        radius_m=0.10,
    )

    exit_result = _exit_result(result)
    assert exit_result["connectivity_class"] == "unknown_bridge_only"
    assert exit_result["strict_connected_candidates"] == []
    assert exit_result["unknown_bridge_candidates"] == ["O01"]


def test_reports_disconnected_when_hard_strip_blocks_both_policies():
    base = np.full((30, 60), FREE_VALUE, dtype=np.uint8)
    base[:, 20:25] = OCCUPIED_VALUE

    result = analyze_handoff_open_area_topology(
        base,
        [_handoff()],
        [_open_region()],
        resolution=0.10,
        radius_m=0.10,
    )

    exit_result = _exit_result(result)
    assert exit_result["connectivity_class"] == "disconnected"
    assert exit_result["strict_connected_candidates"] == []
    assert exit_result["unknown_bridge_candidates"] == []
    assert exit_result["nearest_open_area_label"] == "O01"
    assert exit_result["nearest_open_area_distance_m"] > 0.0
