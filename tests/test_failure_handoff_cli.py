import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, UNKNOWN_VALUE, write_pgm


def _aisle(index, y0):
    return {
        "aisle_id": index,
        "label": f"A{index:02d}",
        "polygon_xy": [[10.0, y0], [110.0, y0], [110.0, y0 + 10.0], [10.0, y0 + 10.0]],
        "centerline_xy": [[10.0, y0 + 5.0], [110.0, y0 + 5.0]],
        "width_m": 1.0,
        "length_m": 10.0,
        "heading_rad": 0.0,
    }


def test_failure_handoff_cli_uses_per_aisle_first_unexpected_radius(tmp_path):
    base = np.full((80, 130), FREE_VALUE, dtype=np.uint8)
    base[15:36, 84:89] = UNKNOWN_VALUE
    base[40:61, 100:105] = UNKNOWN_VALUE
    map_path = tmp_path / "navigation_base_map.pgm"
    write_pgm(base, map_path)

    aisles = [_aisle(3, 20.0), _aisle(10, 45.0)]
    aisle_path = tmp_path / "aisle_rectangles.json"
    aisle_path.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.10,
            "origin": [0.0, 0.0, 0.0],
            "width": 130,
            "height": 80,
        },
        "rectangles": aisles,
    }), encoding="utf-8")

    diagnostics_path = tmp_path / "aisle_geometry_diagnostics.json"
    diagnostics_path.write_text(json.dumps({
        "aisles": [
            {"label": "A03", "first_unexpected_failed_radius_m": 0.20},
            {"label": "A10", "first_unexpected_failed_radius_m": 0.25},
        ]
    }), encoding="utf-8")

    output = tmp_path / "failure_handoffs"
    script = Path(__file__).resolve().parents[1] / "tools" / "estimate_failure_handoffs.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(map_path),
        "--aisles", str(aisle_path),
        "--geometry-diagnostics", str(diagnostics_path),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "failure_handoffs.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["target_count"] == 2
    by_label = {item["label"]: item for item in payload["handoffs"]}
    assert by_label["A03"]["radius_m"] == 0.20
    assert by_label["A10"]["radius_m"] == 0.25
    assert by_label["A03"]["width_clearance_eligible"] is True
    assert by_label["A10"]["width_clearance_eligible"] is True

    # P1-B style localization is the causal diagnosis. The transition-wide
    # source remains contextual and must not silently replace it.
    assert by_label["A03"]["causal_blocker"]["failure_region"] == "exit"
    assert by_label["A03"]["causal_blocker"]["dominant_blocking_source"] == "unknown"
    assert by_label["A10"]["causal_blocker"]["failure_region"] == "exit"
    assert by_label["A10"]["causal_blocker"]["dominant_blocking_source"] == "unknown"
    assert "exit_transition_context_source" in by_label["A03"]
    assert "causal_context_agreement" in by_label["A03"]

    assert "A03: radius=0.20" in completed.stdout
    assert "A10: radius=0.25" in completed.stdout
    assert "causal_source=unknown" in completed.stdout
    assert "exit_context_source=" in completed.stdout
