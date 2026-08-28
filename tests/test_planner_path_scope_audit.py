import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.planner_path_scope_audit import (
    analyze_planner_path_scope_audit,
)


def _aisle(aisle_id, y):
    return {
        "aisle_id": aisle_id,
        "label": f"A{aisle_id:02d}",
        "centerline_xy": [[10.0, float(y)], [50.0, float(y)]],
        "width_m": 4.0,
        "length_m": 40.0,
        "heading_rad": 0.0,
    }


def _profile(shape):
    payload = {
        "grid_shape_yx": list(shape),
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "entry": {
            "bands": [{"mask_key": "entry_depth"}],
            "boundary_uncertainty_mask_key": "entry_boundary",
        },
        "exit": {
            "bands": [{"mask_key": "exit_depth"}],
            "boundary_uncertainty_mask_key": "exit_boundary",
        },
    }
    masks = {
        key: np.zeros(shape, dtype=bool)
        for key in (
            "entry_depth",
            "entry_boundary",
            "exit_depth",
            "exit_boundary",
            "structurally_unresolved_cross",
        )
    }
    masks["entry_depth"][:, :26] = True
    masks["exit_depth"][:, 45:] = True
    return payload, masks


def _connectivity_record(*, connected, first_anchor=(20.0, 15.0), second_anchor=(10.0, 25.0)):
    return {
        "pair_id": "A01-A02",
        "first_aisle": "A01",
        "second_aisle": "A02",
        "side": "entry",
        "radius_m": 0.0,
        "evaluation_status": "evaluated",
        "first_anchor_grid_xy": list(first_anchor),
        "second_anchor_grid_xy": list(second_anchor),
        "baseline_connected": bool(connected),
        "conservative_connected": bool(connected),
    }


def _world(metadata, cells):
    return [list(metadata.grid_to_world(x, y)) for x, y in cells]


def _planner_result(metadata, *, success, path_cells, expected_success=False):
    start = metadata.grid_to_world(20, 15)
    goal = metadata.grid_to_world(10, 25)
    return {
        "request_id": "A01-A02-entry-forward",
        "case_id": "A01-A02-entry",
        "pair_id": "A01-A02",
        "side": "entry",
        "direction": "forward",
        "radius_m": 0.0,
        "expected_success": bool(expected_success),
        "planner_success": bool(success),
        "infrastructure_error": False,
        "start": {"x": start[0], "y": start[1], "yaw": 0.0},
        "goal": {"x": goal[0], "y": goal[1], "yaw": 0.0},
        "path_xy": _world(metadata, path_cells) if success else [],
        "path_length_m": 0.0,
        "failure_reason": None if success else "no valid path",
    }


def _run(base_map, record, result):
    shape = base_map.shape
    profile, masks = _profile(shape)
    metadata = GridMetadata(
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        width=shape[1],
        height=shape[0],
    )
    connectivity = {
        "schema_version": 1,
        "method": "adjacent_aisle_scoped_headland_handoff_connectivity",
        "radius_m": 0.0,
        "pairs": [record],
    }
    planner = {
        "schema_version": 1,
        "method": "nav2_headland_planner_smoke_results",
        "radius_m": 0.0,
        "results": [result],
    }
    return analyze_planner_path_scope_audit(
        base_map,
        [_aisle(1, 15), _aisle(2, 25)],
        connectivity,
        profile,
        masks,
        planner,
        metadata=metadata,
        resolution=1.0,
        radius_m=0.0,
    )["records"][0]


def test_scoped_negative_can_be_explained_by_local_8_connectivity():
    shape = (40, 60)
    base = np.full(shape, 205, dtype=np.uint8)
    diagonal = [(20 - i, 15 + i) for i in range(11)]
    for x, y in diagonal:
        base[y, x] = 254
    metadata = GridMetadata(1.0, 0.0, 0.0, shape[1], shape[0])
    item = _run(
        base,
        _connectivity_record(connected=False),
        _planner_result(metadata, success=True, path_cells=diagonal),
    )

    assert item["strict_connected_4"] is False
    assert item["strict_connected_8"] is True
    assert item["pair_domain_contained"] is True
    assert item["classification"] == "negative_local_8connect_match"


def test_scoped_negative_global_detour_is_not_reported_as_topology_failure():
    shape = (40, 60)
    base = np.full(shape, 254, dtype=np.uint8)
    # Separate the two entry anchors inside the finite pair domain, while
    # leaving the rest of the global map available for an out-of-scope detour.
    base[20, :26] = 205
    metadata = GridMetadata(1.0, 0.0, 0.0, shape[1], shape[0])
    detour = [(20, 15), (30, 10), (40, 10), (50, 20), (30, 30), (10, 25)]
    item = _run(
        base,
        _connectivity_record(connected=False),
        _planner_result(metadata, success=True, path_cells=detour),
    )

    assert item["strict_connected_4"] is False
    assert item["strict4_matches_frozen"] is True
    assert item["planner_success"] is True
    assert item["pair_domain_contained"] is False
    assert item["finite_headland_contained"] is False
    assert item["scope_class"] == "global_outside_finite_headland"
    assert item["classification"] == "negative_global_detour"
    assert item["path_outside_pair_domain_cell_count"] > 0
    assert item["path_outside_pair_domain_fraction"] > 0.0


def test_scoped_negative_no_plan_remains_a_negative_no_plan():
    shape = (40, 60)
    base = np.full(shape, 205, dtype=np.uint8)
    metadata = GridMetadata(1.0, 0.0, 0.0, shape[1], shape[0])
    item = _run(
        base,
        _connectivity_record(connected=False),
        _planner_result(metadata, success=False, path_cells=[]),
    )

    assert item["planner_success"] is False
    assert item["scope_class"] == "no_path"
    assert item["classification"] == "negative_no_plan"
    assert item["path_cell_count"] == 0


def test_positive_local_path_matches_frozen_scoped_connectivity():
    shape = (40, 60)
    base = np.full(shape, 205, dtype=np.uint8)
    local = [(20, 15)] + [(x, 15) for x in range(19, 9, -1)] + [(10, y) for y in range(16, 26)]
    for x, y in local:
        base[y, x] = 254
    metadata = GridMetadata(1.0, 0.0, 0.0, shape[1], shape[0])
    item = _run(
        base,
        _connectivity_record(connected=True),
        _planner_result(metadata, success=True, path_cells=local, expected_success=True),
    )

    assert item["strict_connected_4"] is True
    assert item["strict4_matches_frozen"] is True
    assert item["pair_domain_contained"] is True
    assert item["classification"] == "positive_local_match"


def test_path_audit_reports_source_map_semantics_and_detour_ratio():
    shape = (40, 60)
    base = np.full(shape, 254, dtype=np.uint8)
    base[10, 30] = 205
    base[10, 31] = 0
    metadata = GridMetadata(1.0, 0.0, 0.0, shape[1], shape[0])
    detour = [(20, 15), (30, 10), (31, 10), (10, 25)]
    item = _run(
        base,
        _connectivity_record(connected=False),
        _planner_result(metadata, success=True, path_cells=detour),
    )

    assert item["touches_unknown"] is True
    assert item["touches_occupied"] is True
    assert item["min_source_map_clearance_m"] == 0.0
    assert item["direct_distance_m"] > 0.0
    assert item["path_length_m"] > item["direct_distance_m"]
    assert item["detour_ratio"] > 1.0
