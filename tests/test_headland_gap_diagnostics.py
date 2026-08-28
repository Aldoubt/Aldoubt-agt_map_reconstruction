import numpy as np

from agt_map_reconstruction.maps.headland_gap_diagnostics import (
    _classify_bridge_type,
    analyze_headland_gap_diagnostics,
)


def _aisle(aisle_id, y):
    return {
        "aisle_id": aisle_id,
        "label": f"A{aisle_id:02d}",
        "polygon_xy": [[20.0, y - 4.0], [80.0, y - 4.0], [80.0, y + 4.0], [20.0, y + 4.0]],
        "centerline_xy": [[20.0, float(y)], [80.0, float(y)]],
        "width_m": 1.0,
        "length_m": 6.0,
        "heading_rad": 0.0,
    }


def _fixture(overlay_kind="thin"):
    shape = (60, 100)
    baseline = np.full(shape, 205, dtype=np.uint8)
    baseline[17:24, 20:81] = 254
    baseline[27:34, 20:81] = 254
    conservative = baseline.copy()
    if overlay_kind == "thin":
        conservative[25:26, 8:23] = 254
    elif overlay_kind == "indirect_only":
        conservative[17:24, 19:20] = 254
    elif overlay_kind == "partial_safe":
        conservative[17:25, 8:23] = 254
    elif overlay_kind == "hard_blocked":
        conservative[17:25, 8:23] = 254
        baseline[25:27, 4:31] = 0
        conservative[25:27, 4:31] = 0
    else:
        raise ValueError(overlay_kind)

    aisles = [_aisle(1, 20), _aisle(2, 30)]
    profile = {
        "grid_shape_yx": list(shape),
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "entry": {
            "bands": [{"mask_key": "entry_depth_0_0p5"}],
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
        },
        "exit": {
            "bands": [{"mask_key": "exit_depth_0_0p5"}],
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
        },
    }
    masks = {
        "entry_depth_0_0p5": np.zeros(shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_depth_0_0p5": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_depth_0_0p5"][:, 4:18] = True
    masks["entry_boundary_uncertainty"][:, 18:22] = True
    masks["exit_boundary_uncertainty"][:, 78:82] = True
    masks["exit_depth_0_0p5"][:, 82:96] = True
    connectivity = {
        "radius_m": 0.2,
        "pairs": [
            {
                "pair_id": "A01-A02",
                "first_aisle": "A01",
                "second_aisle": "A02",
                "side": "entry",
                "evaluation_status": "evaluated",
                "first_anchor_grid_xy": [30.0, 20.0],
                "second_anchor_grid_xy": [30.0, 30.0],
                "baseline_connected": False,
                "conservative_connected": False,
            }
        ],
    }
    return baseline, conservative, aisles, profile, masks, connectivity


def _run(kind):
    baseline, conservative, aisles, profile, masks, connectivity = _fixture(kind)
    return analyze_headland_gap_diagnostics(
        baseline,
        conservative,
        aisles,
        connectivity,
        profile,
        masks,
        resolution=0.1,
        radius_m=0.2,
    )["records"][0]


def test_thin_overlay_is_classified_as_eroded_by_clearance():
    item = _run("thin")
    assert item["promoted_free_cell_count_in_domain"] > 0
    assert item["promoted_free_strict_safe_cell_count_in_domain"] == 0
    assert item["baseline_free_newly_safe_cell_count_in_domain"] == 0
    assert item["promoted_free_survival_ratio"] == 0.0
    assert 0.0 < item["max_promoted_free_clearance_m"] < 0.2
    assert item["relaxed_connected"] is True
    assert item["bridge_type"] == "mixed_bridge"
    assert item["failure_class"] == "overlay_eroded_by_clearance"
    assert item["shortest_unknown_bridge_m"] > 0.0


def test_indirect_clearance_gain_is_not_counted_as_promoted_free_survival():
    item = _run("indirect_only")
    assert item["promoted_free_cell_count_in_domain"] > 0
    assert item["promoted_free_strict_safe_cell_count_in_domain"] == 0
    assert item["baseline_free_newly_safe_cell_count_in_domain"] > 0
    assert item["newly_safe_cell_count_in_domain"] == item["baseline_free_newly_safe_cell_count_in_domain"]
    assert item["promoted_free_survival_ratio"] == 0.0
    assert item["failure_class"] == "overlay_indirect_clearance_gain_only"
    assert "new_free_survival_ratio" not in item


def test_safe_overlay_metrics_partition_direct_and_indirect_clearance_gain():
    item = _run("partial_safe")
    assert item["promoted_free_strict_safe_cell_count_in_domain"] > 0
    assert item["baseline_free_newly_safe_cell_count_in_domain"] > 0
    assert item["newly_safe_cell_count_in_domain"] == (
        item["promoted_free_strict_safe_cell_count_in_domain"]
        + item["baseline_free_newly_safe_cell_count_in_domain"]
    )
    assert item["promoted_free_survival_ratio"] == (
        item["promoted_free_strict_safe_cell_count_in_domain"]
        / item["promoted_free_cell_count_in_domain"]
    )
    assert item["max_promoted_free_clearance_m"] >= 0.2
    assert item["failure_class"] == "safe_overlay_not_bridging"


def test_bridge_subtypes_distinguish_unknown_clearance_and_mixed():
    assert _classify_bridge_type({
        "shortest_unknown_bridge_m": 0.2,
        "shortest_non_strict_bridge_m": 0.2,
    }) == "unknown_bridge"
    assert _classify_bridge_type({
        "shortest_unknown_bridge_m": 0.0,
        "shortest_non_strict_bridge_m": 0.2,
    }) == "clearance_only_bridge"
    assert _classify_bridge_type({
        "shortest_unknown_bridge_m": 0.1,
        "shortest_non_strict_bridge_m": 0.3,
    }) == "mixed_bridge"


def _scope_fixture(full_wall=False):
    shape = (60, 100)
    baseline = np.full(shape, 205, dtype=np.uint8)
    conservative = baseline.copy()
    y0, y1 = (0, 60) if full_wall else (13, 38)
    baseline[y0:y1, 15:16] = 0
    conservative[y0:y1, 15:16] = 0

    a1 = _aisle(1, 20)
    a2 = _aisle(2, 30)
    a2["centerline_xy"] = [[10.0, 30.0], [80.0, 30.0]]
    aisles = [a1, a2]
    profile = {
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
        for key in [
            "entry_depth", "entry_boundary", "exit_depth", "exit_boundary",
            "structurally_unresolved_cross",
        ]
    }
    masks["entry_depth"][:, 4:18] = True
    connectivity = {
        "radius_m": 0.2,
        "pairs": [{
            "pair_id": "A01-A02",
            "first_aisle": "A01",
            "second_aisle": "A02",
            "side": "entry",
            "evaluation_status": "evaluated",
            "first_anchor_grid_xy": [30.0, 20.0],
            "second_anchor_grid_xy": [10.0, 30.0],
            "conservative_connected": False,
        }],
    }
    return baseline, conservative, aisles, profile, masks, connectivity


def test_pair_window_scope_blocked_is_separated_from_finite_headland_blocked():
    baseline, conservative, aisles, profile, masks, connectivity = _scope_fixture(False)
    item = analyze_headland_gap_diagnostics(
        baseline, conservative, aisles, connectivity, profile, masks,
        resolution=0.1, radius_m=0.2,
    )["records"][0]
    assert item["relaxed_connected"] is False
    assert item["finite_headland_relaxed_connected"] is True
    assert item["failure_class"] == "pair_window_scope_blocked"
    assert item["pair_window_scope_blocked"] is True
    assert item["hard_or_finite_headland_blocked"] is False

    baseline, conservative, aisles, profile, masks, connectivity = _scope_fixture(True)
    item = analyze_headland_gap_diagnostics(
        baseline, conservative, aisles, connectivity, profile, masks,
        resolution=0.1, radius_m=0.2,
    )["records"][0]
    assert item["relaxed_connected"] is False
    assert item["finite_headland_relaxed_connected"] is False
    assert item["failure_class"] == "hard_or_finite_headland_blocked"
    assert item["pair_window_scope_blocked"] is False
    assert item["hard_or_finite_headland_blocked"] is True


def test_hard_wall_is_not_overstated_as_proven_hard_only():
    item = _run("hard_blocked")
    assert item["relaxed_connected"] is False
    assert item["finite_headland_relaxed_connected"] is False
    assert item["failure_class"] == "hard_or_finite_headland_blocked"
    assert item["hard_or_finite_headland_blocked"] is True
    assert item["shortest_unknown_bridge_m"] is None
