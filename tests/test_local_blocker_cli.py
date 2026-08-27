import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    write_pgm,
)


def test_local_blocker_cli_writes_json_and_csv(tmp_path):
    base = np.full((30, 100), UNKNOWN_VALUE, dtype=np.uint8)
    base[10:20, 5:85] = FREE_VALUE
    base[10:20, 44:47] = OCCUPIED_VALUE

    map_path = tmp_path / "navigation_base_map.pgm"
    write_pgm(base, map_path)

    aisle = {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[5, 10], [84, 10], [84, 19], [5, 19]],
        "centerline_xy": [[5, 14.5], [84, 14.5]],
        "width_m": 0.50,
        "length_m": 4.00,
    }
    aisle_path = tmp_path / "aisle_rectangles.json"
    aisle_path.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.05,
            "origin": [0.0, 0.0, 0.0],
            "width": 100,
            "height": 30,
        },
        "rectangles": [aisle],
    }), encoding="utf-8")

    diagnostics_path = tmp_path / "aisle_geometry_diagnostics.json"
    diagnostics_path.write_text(json.dumps({
        "aisles": [{
            "label": "A01",
            "first_unexpected_failed_radius_m": 0.20,
        }]
    }), encoding="utf-8")

    output = tmp_path / "blockers"
    script = Path(__file__).resolve().parents[1] / "tools" / "localize_aisle_blockers.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(map_path),
        "--aisles", str(aisle_path),
        "--geometry-diagnostics", str(diagnostics_path),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "blocker_localization.json").read_text())
    assert payload["target_count"] == 1
    assert payload["targets"][0]["label"] == "A01"
    assert payload["targets"][0]["failure_region"] == "interior"
    assert payload["targets"][0]["dominant_blocking_source"] == "hard"
    assert (output / "blocker_localization.csv").is_file()
    assert "A01" in completed.stdout
