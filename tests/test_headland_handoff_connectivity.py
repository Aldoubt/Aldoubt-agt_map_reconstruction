import json

import numpy as np

from agt_map_reconstruction.maps.headland_handoff_connectivity import (
    analyze_headland_handoff_connectivity,
    build_planner_pairs,
    write_headland_connectivity_bundle,
)


def _aisle(aisle_id, y, width_m=1.0):
    return {
        "aisle_id": aisle_id,
        "label": f"A{aisle_id:02d}",
        "polygon_xy": [[20.0, y - 4.0], [80.0, y - 4.0], [80.0, y + 4.0], [20.0, y + 4.0]],
        "centerline_xy": [[20.0, float(y)], [80.0, float(y)]],
        "width_m": float(width_m),
        "length_m": 6.0,
        "heading_rad": 0.0,
    }


def _handoff(aisle, entry_xy, exit_xy, width_ok=True):
    return {
        "aisle_id": aisle["aisle_id"],
        "label": aisle["label"],
        "status": "ok",
        "width_clearance_eligible": bool(width_ok),
        "entry_handoff": {
            "grid_xy": list(map(float, entry_xy)),
            "map_xy_m": [0.1 * float(entry_xy[0]), 0.1 * float(entry_xy[1])],
            "heading_rad": 0.0,
            "clearance_m": 0.3,
        },
        "exit_handoff": {
            "grid_xy": list(map(float, exit_xy)),
            "map_xy_m": [0.1 * float(exit_xy[0]), 0.1 * float(exit_xy[1])],
            "heading_rad": 0.0,
            "clearance_m": 0.3,
        },
    }


def _depth_profile(shape):
    masks = {}
    for name in (
        "entry_depth_0_0p5",
        "entry_boundary_uncertainty",
        "exit_depth_0_0p5",
        "exit_boundary_uncertainty",
        "structurally_unresolved_cross",
    ):
        masks[name] = np.zeros(shape, dtype=bool)
    masks["entry_depth_0_0p5"][:, 4:18] = True
    masks["entry_boundary_uncertainty"][:, 18:22] = True
    masks["exit_boundary_uncertainty"][:, 78:82] = True
    masks["exit_depth_0_0p5"][:, 82:96] = True
    payload = {
        "grid_shape_yx": list(shape),
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "entry": {
            "bands": [{"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
        },
        "exit": {
            "bands": [{"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
        },
    }
    return payload, masks


def _maps_and_handoffs(width2=1.0):
    shape = (60, 100)
    unknown, free = 205, 254
    baseline = np.full(shape, unknown, dtype=np.uint8)
    baseline[17:24, 20:81] = free
    baseline[27:34, 20:81] = free
    conservative = baseline.copy()
    conservative[17:34, 8:23] = free

    a1 = _aisle(1, 20)
    a2 = _aisle(2, 30, width_m=width2)
    aisles = [a1, a2]
    baseline_handoffs = [
        _handoff(a1, (30, 20), (70, 20), True),
        _handoff(a2, (30, 30), (70, 30), width2 >= 0.4),
    ]
    conservative_handoffs = [
        _handoff(a1, (24, 20), (70, 20), True),
        _handoff(a2, (24, 30), (70, 30), width2 >= 0.4),
    ]
    payload, masks = _depth_profile(shape)
    return baseline, conservative, aisles, baseline_handoffs, conservative_handoffs, payload, masks


def test_entry_pair_gain_is_detected_without_leaking_to_exit():
    baseline, conservative, aisles, bh, ch, payload, masks = _maps_and_handoffs()
    result = analyze_headland_handoff_connectivity(
        baseline, conservative, aisles, bh, ch, payload, masks,
        resolution=0.1, radius_m=0.2,
    )

    assert result["adjacent_pair_count"] == 1
    entry = next(item for item in result["pairs"] if item["side"] == "entry")
    exit_ = next(item for item in result["pairs"] if item["side"] == "exit")
    assert entry["pair_id"] == "A01-A02"
    assert entry["baseline_connected"] is False
    assert entry["conservative_connected"] is True
    assert entry["gained_by_trusted_overlay"] is True
    assert entry["lost_by_trusted_overlay"] is False
    assert entry["new_free_cell_count_in_domain"] > 0
    assert exit_["baseline_connected"] is False
    assert exit_["conservative_connected"] is False
    assert exit_["gained_by_trusted_overlay"] is False


def test_width_ineligible_adjacent_pair_is_not_accepted():
    baseline, conservative, aisles, bh, ch, payload, masks = _maps_and_handoffs(width2=0.3)
    result = analyze_headland_handoff_connectivity(
        baseline, conservative, aisles, bh, ch, payload, masks,
        resolution=0.1, radius_m=0.2,
    )
    for item in result["pairs"]:
        assert item["evaluation_status"] == "width_ineligible"
        assert item["conservative_connected"] is False


def test_only_consecutive_aisles_form_pairs():
    shape = (60, 100)
    unknown, free = 205, 254
    baseline = np.full(shape, unknown, dtype=np.uint8)
    conservative = baseline.copy()
    payload, masks = _depth_profile(shape)
    aisles = [_aisle(1, 15), _aisle(2, 25), _aisle(3, 35)]
    for aisle in aisles:
        y = int(aisle["centerline_xy"][0][1])
        baseline[y-3:y+4, 20:81] = free
        conservative[y-3:y+4, 20:81] = free
    handoffs = [
        _handoff(aisle, (30, aisle["centerline_xy"][0][1]), (70, aisle["centerline_xy"][0][1]))
        for aisle in aisles
    ]
    result = analyze_headland_handoff_connectivity(
        baseline, conservative, aisles, handoffs, handoffs, payload, masks,
        resolution=0.1, radius_m=0.2,
    )
    assert result["adjacent_pair_count"] == 2
    assert sorted(set(item["pair_id"] for item in result["pairs"])) == ["A01-A02", "A02-A03"]
    assert all(item["pair_id"] != "A01-A03" for item in result["pairs"])


def test_planner_pairs_use_outward_to_inward_headings():
    baseline, conservative, aisles, bh, ch, payload, masks = _maps_and_handoffs()
    result = analyze_headland_handoff_connectivity(
        baseline, conservative, aisles, bh, ch, payload, masks,
        resolution=0.1, radius_m=0.2,
    )
    planner = build_planner_pairs(result)
    entry = next(item for item in planner["tests"] if item["side"] == "entry")
    assert entry["enabled"] is True
    assert np.isclose(abs(entry["forward"]["start"]["yaw"]), np.pi)
    assert np.isclose(entry["forward"]["goal"]["yaw"], 0.0)
    assert entry["reverse"]["start"] is not None
    assert entry["reverse"]["goal"] is not None


def test_bundle_writes_frozen_outputs(tmp_path):
    baseline, conservative, aisles, bh, ch, payload, masks = _maps_and_handoffs()
    result = analyze_headland_handoff_connectivity(
        baseline, conservative, aisles, bh, ch, payload, masks,
        resolution=0.1, radius_m=0.2,
    )
    write_headland_connectivity_bundle(
        result, conservative, tmp_path,
        source_map_yaml="navigation_conservative_v2/navigation_base_map.yaml",
    )
    expected = {
        "headland_connectivity.json",
        "headland_gates.geojson",
        "planner_pairs.yaml",
        "headland_connectivity.png",
    }
    assert expected == {p.name for p in tmp_path.iterdir()}
    payload = json.loads((tmp_path / "headland_connectivity.json").read_text())
    assert payload["adjacent_pair_count"] == 1
