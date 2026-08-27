import json
from pathlib import Path
import subprocess
import sys


def _row(label, y):
    return {
        "label": label,
        "region_class": "row_aisle",
        "centerline_xy": [[20.0, float(y)], [80.0, float(y)]],
        "polygon_xy": [[20.0, y - 2.0], [80.0, y - 2.0], [80.0, y + 2.0], [20.0, y + 2.0]],
    }


def test_headland_candidate_geometry_cli_writes_audit(tmp_path):
    regions = [
        _row("A01", 10),
        _row("A02", 20),
        _row("A03", 30),
        {
            "label": "O01",
            "region_class": "wide_open_area_candidate",
            "source_band_label": "A18",
            "polygon_xy": [[20.0, 40.0], [80.0, 40.0], [80.0, 45.0], [20.0, 45.0]],
            "centerline_xy": [[20.0, 42.5], [80.0, 42.5]],
        },
    ]
    source = tmp_path / "row_band_regions.json"
    source.write_text(json.dumps({
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
            "width": 110,
            "height": 60,
        },
        "regions": regions,
    }), encoding="utf-8")

    output = tmp_path / "audit"
    script = Path(__file__).resolve().parents[1] / "tools" / "audit_headland_candidate_geometry.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--row-band-regions", str(source),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "headland_candidate_geometry.json").read_text())
    assert payload["row_aisle_count"] == 3
    assert payload["candidate_count"] == 1
    item = payload["candidates"][0]
    assert item["label"] == "O01"
    assert item["row_axis_alignment"] > 0.95
    assert item["cross_row_overlap_fraction"] < 0.05
    assert item["semantic_promotion"] is False
    assert (output / "headland_candidate_geometry.csv").is_file()
    assert "O01:" in completed.stdout
