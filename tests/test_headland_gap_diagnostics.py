import numpy as np

from agt_map_reconstruction.maps.headland_gap_diagnostics import analyze_headland_gap_diagnostics


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
                "new_free_cell_count_in_domain": 1,
                "new_safe_cell_count_in_domain": 0,
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
    assert item["new_free_cell_count_in_domain"] > 0
    assert item["new_safe_cell_count_in_domain"] == 0
    assert item["new_free_survival_ratio"] == 0.0
    assert 0.0 < item["max_new_free_clearance_m"] < 0.2
    assert item["relaxed_connected"] is True
    assert item["bridge_class"] == "unknown_bridge_only"
    assert item["failure_class"] == "overlay_eroded_by_clearance"
    assert item["shortest_unknown_bridge_m"] > 0.0


def test_safe_overlay_without_full_bridge_reports_remaining_unknown_gap():
    item = _run("partial_safe")
    assert item["new_safe_cell_count_in_domain"] > 0
    assert item["new_free_survival_ratio"] > 0.0
    assert item["relaxed_connected"] is True
    assert item["bridge_class"] == "unknown_bridge_only"
    assert item["failure_class"] == "safe_overlay_not_bridging"
    assert item["shortest_unknown_bridge_m"] > 0.0
    assert item["shortest_unknown_bridge_cell_count"] > 0
    assert item["shortest_non_strict_bridge_m"] >= item["shortest_unknown_bridge_m"]


def test_hard_wall_is_reported_as_hard_or_scope_blocked():
    item = _run("hard_blocked")
    assert item["relaxed_connected"] is False
    assert item["hard_blocked"] is True
    assert item["bridge_class"] == "hard_or_scope_blocked"
    assert item["failure_class"] == "hard_or_scope_blocked"
    assert item["shortest_unknown_bridge_m"] is None
