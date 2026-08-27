import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, OCCUPIED_VALUE


def _script():
    return Path(__file__).resolve().parents[1] / "tools" / "build_structural_endpoint_d31.py"


def test_structural_endpoint_cli_help_exposes_all_explicit_parameters():
    completed = subprocess.run(
        [sys.executable, str(_script()), "--help"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    for flag in (
        "--map",
        "--row-band-regions",
        "--handoffs",
        "--strip-width-m",
        "--bin-size-m",
        "--min-support-fraction",
        "--min-persistence-m",
        "--max-internal-gap-m",
        "--max-side-endpoint-disagreement-m",
        "--residual-floor-m",
        "--mad-scale",
        "--min-inlier-count",
    ):
        assert flag in completed.stdout


def test_structural_endpoint_cli_keeps_geometric_handoff_and_structural_semantics_separate(tmp_path):
    base = np.full((60, 80), FREE_VALUE, dtype=np.uint8)
    rows = []
    handoffs = []
    for index, y in enumerate((12.0, 30.0, 48.0), start=1):
        label = f"A{index:02d}"
        rows.append(
            {
                "label": label,
                "region_class": "row_aisle",
                "polygon_xy": [[5.0, y - 2.0], [70.0, y - 2.0], [70.0, y + 2.0], [5.0, y + 2.0]],
                "centerline_xy": [[5.0, y], [70.0, y]],
            }
        )
        # Structural strips terminate well before the geometric aisle endpoint.
        yi = int(y)
        base[yi - 6 : yi - 2, 10:50] = OCCUPIED_VALUE
        base[yi + 3 : yi + 7, 10:50] = OCCUPIED_VALUE
        handoffs.append(
            {
                "label": label,
                "status": "ok",
                "width_clearance_eligible": True,
                "entry_handoff": {"grid_xy": [6.0, y], "clearance_m": 0.2},
                "exit_handoff": {"grid_xy": [69.0, y], "clearance_m": 0.2},
            }
        )

    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.flipud(base))
    regions_path = tmp_path / "regions.json"
    regions_path.write_text(
        json.dumps(
            {
                "grid": {
                    "resolution": 0.10,
                    "width": 80,
                    "height": 60,
                    "origin": [0.0, 0.0, 0.0],
                    "frame_id": "map",
                },
                "regions": rows,
            }
        ),
        encoding="utf-8",
    )
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_path.write_text(
        json.dumps({"radius_m": 0.20, "handoffs": handoffs}),
        encoding="utf-8",
    )
    output = tmp_path / "d31"

    completed = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--map",
            str(map_path),
            "--row-band-regions",
            str(regions_path),
            "--handoffs",
            str(handoffs_path),
            "--output",
            str(output),
            "--strip-width-m",
            "0.40",
            "--bin-size-m",
            "0.50",
            "--min-support-fraction",
            "0.50",
            "--min-persistence-m",
            "2.0",
            "--max-internal-gap-m",
            "0.50",
            "--max-side-endpoint-disagreement-m",
            "0.75",
            "--residual-floor-m",
            "0.30",
            "--mad-scale",
            "3.0",
            "--min-inlier-count",
            "3",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    for name in (
        "structural_endpoint_profiles.json",
        "structural_endpoint_boundary.json",
        "structural_endpoint_context.png",
        "entry_structural_endpoint_context.png",
        "exit_structural_endpoint_context.png",
    ):
        assert (output / name).exists()

    payload = json.loads((output / "structural_endpoint_boundary.json").read_text())
    assert payload["policy"]["navigation_map_modified"] is False
    assert payload["policy"]["semantic_promotion"] is False
    assert payload["policy"]["automatic_acceptance"] is False
    assert len(payload["rows"]) == 3
    for row in payload["rows"]:
        assert row["raw_geometric_entry_grid_xy"] is not None
        assert row["raw_geometric_exit_grid_xy"] is not None
        assert row["clearance_handoff_entry_grid_xy"] is not None
        assert row["clearance_handoff_exit_grid_xy"] is not None
        assert row["entry"]["status"] == "ok_bilateral"
        assert row["exit"]["status"] == "ok_bilateral"
        assert row["entry"]["structural_grid_xy"] is not None
        assert row["exit"]["structural_grid_xy"] is not None

    assert payload["robust_boundary"]["entry"]["fit_status"] == "ok"
    assert payload["robust_boundary"]["exit"]["fit_status"] == "ok"
