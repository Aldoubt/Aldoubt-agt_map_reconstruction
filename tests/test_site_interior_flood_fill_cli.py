import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import OCCUPIED_VALUE, UNKNOWN_VALUE


def _write_closed_room(path, *, gap=False):
    grid = np.full((7, 7), UNKNOWN_VALUE, dtype=np.uint8)
    grid[1, 1:6] = OCCUPIED_VALUE
    grid[5, 1:6] = OCCUPIED_VALUE
    grid[1:6, 1] = OCCUPIED_VALUE
    grid[1:6, 5] = OCCUPIED_VALUE
    if gap:
        grid[1, 3] = UNKNOWN_VALUE
    cv2.imwrite(str(path), np.flipud(grid))


def _write_lattice(path):
    path.write_text(
        json.dumps(
            {
                "slots": [
                    {
                        "slot_id": "L01",
                        "source": "observed_row_aisle",
                        "centerline_xy": [[2.0, 3.0], [4.0, 3.0]],
                    },
                    {
                        "slot_id": "L02",
                        "source": "lattice_inferred_wide_band",
                        "centerline_xy": [[2.0, 4.0], [4.0, 4.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_site_interior_flood_fill_cli_writes_anchor_validated_enclosed_mask(tmp_path):
    map_path = tmp_path / "map.pgm"
    lattice_path = tmp_path / "row_lattice_completion.json"
    _write_closed_room(map_path)
    _write_lattice(lattice_path)
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_site_interior_flood_fill.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map",
            str(map_path),
            "--row-lattice-completion",
            str(lattice_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "site_interior_flood_fill.json").read_text(encoding="utf-8"))
    mask = np.load(output / "site_interior_nonhard_mask.npy")
    anchors = np.load(output / "interior_anchor_mask.npy")
    assert payload["status"] == "ok"
    assert payload["interior_anchor_validation_requested"] is True
    assert payload["interior_anchor_validation_passed"] is True
    assert payload["interior_anchor_cell_count"] == 1
    assert payload["interior_nonhard_cell_count"] == 9
    assert int(np.count_nonzero(mask)) == 9
    assert int(np.count_nonzero(anchors)) == 1
    assert (output / "site_interior_flood_fill.png").exists()
    assert "interior_anchor_validation_passed: true" in completed.stdout
    assert "morphology_applied: false" in completed.stdout


def test_site_interior_flood_fill_cli_reports_anchor_reachable_wall_gap_failure(tmp_path):
    map_path = tmp_path / "map.pgm"
    lattice_path = tmp_path / "row_lattice_completion.json"
    _write_closed_room(map_path, gap=True)
    _write_lattice(lattice_path)
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_site_interior_flood_fill.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map",
            str(map_path),
            "--row-lattice-completion",
            str(lattice_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    payload = json.loads((output / "site_interior_flood_fill.json").read_text(encoding="utf-8"))
    leak_path = np.load(output / "leak_path_mask.npy")
    assert payload["status"] == "leaked_or_unenclosed"
    assert payload["interior_anchor_validation_passed"] is False
    assert payload["interior_anchor_exterior_reachable_cell_count"] == 1
    assert int(np.count_nonzero(leak_path)) > 0
    assert payload["automatic_wall_gap_closure"] is False
    assert "leak_anchor_xy:" in completed.stdout
    assert "leaked_or_unenclosed" in completed.stdout
