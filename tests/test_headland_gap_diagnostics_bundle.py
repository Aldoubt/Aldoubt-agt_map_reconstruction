import json

import numpy as np

from agt_map_reconstruction.maps.headland_gap_diagnostics import (
    write_headland_gap_diagnostics_bundle,
)


def test_bundle_writes_clarified_metric_semantics(tmp_path):
    base = np.full((20, 30), 205, dtype=np.uint8)
    base[5:15, 5:25] = 254
    diagnostics = {
        "schema_version": 2,
        "method": "scoped_headland_gap_diagnostics",
        "radius_m": 0.2,
        "records": [
            {
                "pair_id": "A09-A10",
                "side": "entry",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "relaxed_connected": True,
                "finite_headland_relaxed_connected": True,
                "bridge_type": "mixed_bridge",
                "failure_class": "safe_overlay_not_bridging",
                "pair_window_scope_blocked": False,
                "hard_or_finite_headland_blocked": False,
                "promoted_free_cell_count_in_domain": 56,
                "promoted_free_strict_safe_cell_count_in_domain": 0,
                "baseline_free_newly_safe_cell_count_in_domain": 6,
                "newly_safe_cell_count_in_domain": 6,
                "promoted_free_survival_ratio": 0.0,
                "max_promoted_free_clearance_m": 0.10,
                "shortest_unknown_bridge_m": 0.35,
                "shortest_unknown_bridge_cell_count": 4,
                "shortest_non_strict_bridge_m": 0.45,
                "relaxed_path_length_m": 1.2,
                "first_anchor_grid_xy": [6.0, 8.0],
                "second_anchor_grid_xy": [20.0, 12.0],
                "bridge_path_grid_xy": [[6, 8], [7, 9], [8, 10], [9, 11]],
            }
        ],
    }

    outputs = write_headland_gap_diagnostics_bundle(diagnostics, base, tmp_path)
    expected = {
        "headland_gap_diagnostics.json",
        "headland_gap_diagnostics.csv",
        "headland_gap_diagnostics.png",
    }
    assert expected == {p.name for p in tmp_path.iterdir()}
    payload = json.loads((tmp_path / "headland_gap_diagnostics.json").read_text())
    assert payload["records"][0]["bridge_type"] == "mixed_bridge"
    csv_text = (tmp_path / "headland_gap_diagnostics.csv").read_text()
    for field in [
        "promoted_free_strict_safe_cell_count_in_domain",
        "baseline_free_newly_safe_cell_count_in_domain",
        "finite_headland_relaxed_connected",
        "bridge_type",
        "hard_or_finite_headland_blocked",
    ]:
        assert field in csv_text
    assert "new_free_survival_ratio" not in csv_text
    assert (tmp_path / "headland_gap_diagnostics.png").stat().st_size > 0
    assert set(outputs) == {"json", "csv", "overlay"}
