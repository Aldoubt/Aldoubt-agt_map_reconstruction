import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, write_pgm


def test_open_area_topology_cli_writes_json_and_csv(tmp_path):
    base = np.full((30, 60), FREE_VALUE, dtype=np.uint8)
    map_path = tmp_path / "navigation_base_map.pgm"
    write_pgm(base, map_path)

    regions_path = tmp_path / "row_band_regions.json"
    regions_path.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.10,
            "origin": [0.0, 0.0, 0.0],
            "width": 60,
            "height": 30,
        },
        "regions": [{
            "label": "O01",
            "region_class": "wide_open_area_candidate",
            "polygon_xy": [[30.0, 10.0], [45.0, 10.0], [45.0, 20.0], [30.0, 20.0]],
        }],
    }), encoding="utf-8")

    handoffs_path = tmp_path / "aisle_handoffs.json"
    handoffs_path.write_text(json.dumps({
        "schema_version": 2,
        "radius_m": 0.10,
        "handoffs": [{
            "label": "A01",
            "status": "ok",
            "width_clearance_eligible": True,
            "entry_handoff": {"grid_xy": [5.0, 15.0]},
            "exit_handoff": {"grid_xy": [10.0, 15.0]},
        }],
    }), encoding="utf-8")

    output = tmp_path / "topology"
    script = Path(__file__).resolve().parents[1] / "tools" / "analyze_open_area_topology.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(map_path),
        "--row-band-regions", str(regions_path),
        "--handoffs", str(handoffs_path),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "open_area_topology.json").read_text())
    assert payload["handoff_count"] == 2
    assert payload["open_area_candidate_count"] == 1
    assert payload["connectivity_counts"]["strict_connected"] == 2
    assert payload["open_area_candidates"][0]["label"] == "O01"
    assert payload["open_area_candidates"][0]["semantic_promotion"] is False
    assert (output / "open_area_topology.csv").is_file()
    assert "open_area_candidates: 1" in completed.stdout
    assert "O01:" in completed.stdout
