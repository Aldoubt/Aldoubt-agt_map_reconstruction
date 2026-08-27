import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, write_pgm


def test_headland_endpoint_envelope_cli_writes_json(tmp_path):
    base = np.full((50, 70), FREE_VALUE, dtype=np.uint8)
    map_path = tmp_path / "navigation_base_map.pgm"
    write_pgm(base, map_path)

    regions_path = tmp_path / "row_band_regions.json"
    regions_path.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.10,
            "origin": [0.0, 0.0, 0.0],
            "width": 70,
            "height": 50,
        },
        "regions": [
            {
                "label": "A01",
                "region_class": "row_aisle",
                "polygon_xy": [[10.0, 10.0], [40.0, 10.0], [40.0, 16.0], [10.0, 16.0]],
                "centerline_xy": [[10.0, 13.0], [40.0, 13.0]],
            },
            {
                "label": "A02",
                "region_class": "row_aisle",
                "polygon_xy": [[10.0, 20.0], [40.0, 20.0], [40.0, 26.0], [10.0, 26.0]],
                "centerline_xy": [[10.0, 23.0], [40.0, 23.0]],
            },
        ],
    }), encoding="utf-8")

    handoffs_path = tmp_path / "aisle_handoffs.json"
    handoffs_path.write_text(json.dumps({
        "schema_version": 2,
        "radius_m": 0.10,
        "handoffs": [
            {
                "label": "A01",
                "status": "ok",
                "width_clearance_eligible": True,
                "entry_handoff": {"grid_xy": [11.0, 13.0]},
                "exit_handoff": {"grid_xy": [39.0, 13.0]},
            },
            {
                "label": "A02",
                "status": "ok",
                "width_clearance_eligible": True,
                "entry_handoff": {"grid_xy": [11.0, 23.0]},
                "exit_handoff": {"grid_xy": [39.0, 23.0]},
            },
        ],
    }), encoding="utf-8")

    output = tmp_path / "endpoint_envelope"
    script = Path(__file__).resolve().parents[1] / "tools" / "analyze_headland_endpoint_envelope.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(map_path),
        "--row-band-regions", str(regions_path),
        "--handoffs", str(handoffs_path),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "headland_endpoint_envelope.json").read_text())
    assert payload["eligible_row_count"] == 2
    assert payload["policy"]["semantic_promotion"] is False
    assert payload["sides"]["entry"]["strict"]["best_component"] is not None
    assert payload["sides"]["exit"]["strict"]["best_component"] is not None
    assert "eligible_rows: 2" in completed.stdout
    assert "entry:" in completed.stdout
    assert "exit:" in completed.stdout
