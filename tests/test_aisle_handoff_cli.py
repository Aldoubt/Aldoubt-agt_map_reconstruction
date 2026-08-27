import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    UNKNOWN_VALUE,
    write_pgm,
)


def test_aisle_handoff_cli_writes_clearance_conditioned_asset(tmp_path):
    base = np.full((60, 130), FREE_VALUE, dtype=np.uint8)
    base[15:36, 84:89] = UNKNOWN_VALUE
    map_path = tmp_path / "navigation_base_map.pgm"
    write_pgm(base, map_path)

    aisle = {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[10.0, 20.0], [110.0, 20.0], [110.0, 30.0], [10.0, 30.0]],
        "centerline_xy": [[10.0, 25.0], [110.0, 25.0]],
        "centerline_map_xy_m": [[1.0, 2.0], [11.0, 2.0]],
        "width_m": 1.0,
        "length_m": 10.0,
        "heading_rad": 0.0,
    }
    aisle_path = tmp_path / "aisle_rectangles.json"
    aisle_path.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.10,
            "origin": [0.0, 0.0, 0.0],
            "width": 130,
            "height": 60,
        },
        "rectangles": [aisle],
    }), encoding="utf-8")

    output = tmp_path / "handoffs"
    script = Path(__file__).resolve().parents[1] / "tools" / "estimate_aisle_handoffs.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(map_path),
        "--aisles", str(aisle_path),
        "--output", str(output),
        "--radius", "0.20",
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "aisle_handoffs.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["radius_m"] == 0.20
    assert payload["aisle_count"] == 1
    assert payload["ok_count"] == 1
    handoff = payload["handoffs"][0]
    assert handoff["label"] == "A01"
    assert handoff["exit_handoff"]["boundary_nearest_source"] == "unknown"
    assert handoff["exit_transition"]["dominant_source"] == "unknown"
    assert handoff["exit_transition_length_m"] > 2.5
    assert (output / "aisle_handoffs.csv").is_file()
    assert "radius: 0.20" in completed.stdout
    assert "exit_transition_source=unknown" in completed.stdout
    assert "A01" in completed.stdout
