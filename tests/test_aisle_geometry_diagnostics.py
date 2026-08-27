import json
from pathlib import Path
import subprocess
import sys

from agt_map_reconstruction.maps.aisle_geometry_diagnostics import (
    diagnose_aisle_geometry,
)


def _validation():
    widths = [
        0.35, 0.70, 0.70, 0.55, 0.65,
        0.40, 0.35, 0.75, 0.60, 0.70,
        0.80, 0.50, 0.75, 0.95, 0.80,
        1.00, 0.50, 1.75, 2.90, 4.65,
    ]
    radii = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    aisles = []
    for index, width in enumerate(widths, start=1):
        aisles.append({
            "aisle_id": index,
            "label": f"A{index:02d}",
            "width_m": width,
            "length_m": 30.0,
            "clearance_pass": {
                f"{radius:.2f}": radius <= width / 2.0 + 1e-12
                for radius in radii
            },
        })

    # Inject connectivity defects beyond the width-only model.
    for index in (3, 12, 14):
        for key in aisles[index - 1]["clearance_pass"]:
            aisles[index - 1]["clearance_pass"][key] = False
    for key in ("0.25", "0.30", "0.35"):
        aisles[9]["clearance_pass"][key] = False

    return {
        "resolution_m": 0.05,
        "clearance_tests": {
            f"{radius:.2f}": {"radius_m": radius}
            for radius in radii
        },
        "aisles": aisles,
    }


def test_diagnosis_separates_width_connectivity_and_wide_outliers():
    result = diagnose_aisle_geometry(_validation())

    assert result["summary"]["minimum_width_limited"] == ["A01", "A07"]
    assert result["summary"]["minimum_connectivity_limited"] == [
        "A03", "A12", "A14"
    ]
    assert result["summary"]["unexpected_connectivity_failures"] == [
        "A03", "A10", "A12", "A14"
    ]
    assert result["summary"]["wide_width_outliers"] == [
        "A18", "A19", "A20"
    ]
    assert abs(
        result["summary"]["wide_width_outlier_threshold_m"] - 1.2875
    ) < 1e-9


def test_cli_writes_json_and_csv(tmp_path):
    source = tmp_path / "validation.json"
    source.write_text(json.dumps(_validation()), encoding="utf-8")
    output = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "diagnose_aisle_geometry.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validation", str(source),
            "--output", str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "aisle_geometry_diagnostics.json").is_file()
    assert (output / "aisle_geometry_diagnostics.csv").is_file()
