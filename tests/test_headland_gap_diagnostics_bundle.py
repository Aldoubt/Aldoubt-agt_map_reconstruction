import json
from pathlib import Path

import numpy as np

from agt_map_reconstruction.maps.headland_gap_diagnostics import (
    write_headland_gap_diagnostics_bundle,
)


def test_bundle_writes_json_csv_and_overlay(tmp_path):
    base = np.full((20, 30), 205, dtype=np.uint8)
    base[5:15, 5:25] = 254
    diagnostics = {
        "schema_version": 1,
        "method": "scoped_headland_gap_diagnostics",
        "radius_m": 0.2,
        "records": [
            {
                "pair_id": "A09-A10",
                "side": "entry",
                "evaluation_status": "evaluated",
                "strict_connected": False,
                "relaxed_connected": True,
                "bridge_class": "unknown_bridge_only",
                "failure_class": "safe_overlay_not_bridging",
                "new_free_cell_count_in_domain": 56,
                "new_safe_cell_count_in_domain": 6,
                "new_free_survival_ratio": 6 / 56,
                "max_new_free_clearance_m": 0.25,
                "shortest_unknown_bridge_m": 0.35,
                "shortest_unknown_bridge_cell_count": 4,
                "shortest_non_strict_bridge_m": 0.45,
                "relaxed_path_length_m": 1.2,
                "hard_blocked": False,
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
    assert payload["records"][0]["pair_id"] == "A09-A10"
    csv_text = (tmp_path / "headland_gap_diagnostics.csv").read_text()
    assert "shortest_unknown_bridge_m" in csv_text
    assert "A09-A10" in csv_text
    assert (tmp_path / "headland_gap_diagnostics.png").stat().st_size > 0
    assert set(outputs) == {"json", "csv", "overlay"}
