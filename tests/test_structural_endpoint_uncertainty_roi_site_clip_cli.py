import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE


def _write_roi(tmp_path, shape, map_path):
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    masks = {
        "entry_conservative_outward": np.zeros(shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_conservative_outward": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_conservative_outward"][1, 0:3] = True
    masks["entry_boundary_uncertainty"][1, 3] = True
    masks["exit_boundary_uncertainty"][2, 4] = True
    masks["exit_conservative_outward"][2, 5:8] = True
    masks["structurally_unresolved_cross"][3, 1:7] = True
    mask_files = {}
    for name, mask in masks.items():
        filename = f"{name}.npy"
        np.save(roi_dir / filename, mask)
        mask_files[name] = filename
    roi_path = roi_dir / "structural_endpoint_uncertainty_roi.json"
    roi_path.write_text(
        json.dumps(
            {
                "method": "fused_structural_endpoint_uncertainty_roi",
                "uncertainty_quantile": "p95",
                "unresolved_ridge_ids": ["R_X"],
                "mask_files": mask_files,
                "sources": {"map": str(map_path)},
            }
        ),
        encoding="utf-8",
    )
    return roi_path


def _write_site(tmp_path, shape, map_path, *, anchor_validated):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    site = np.zeros(shape, dtype=bool)
    site[1:4, 2:6] = True
    exterior = ~site
    np.save(site_dir / "site_interior_nonhard_mask.npy", site)
    np.save(site_dir / "exterior_reachable_nonhard_mask.npy", exterior)
    site_path = site_dir / "site_interior_flood_fill.json"
    payload = {
        "schema_version": 2 if anchor_validated else 1,
        "status": "ok",
        "mask_files": {
            "site_interior_nonhard": "site_interior_nonhard_mask.npy",
            "exterior_reachable_nonhard": "exterior_reachable_nonhard_mask.npy",
        },
        "sources": {"map": str(map_path)},
    }
    if anchor_validated:
        payload.update(
            {
                "status_basis": "trusted_interior_anchor_validation",
                "interior_anchor_validation_requested": True,
                "interior_anchor_validation_passed": True,
                "interior_anchor_exterior_reachable_cell_count": 0,
            }
        )
    site_path.write_text(json.dumps(payload), encoding="utf-8")
    return site_path


def test_site_clip_cli_preserves_roi_interface_and_removes_exterior(tmp_path):
    shape = (5, 8)
    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.flipud(np.full(shape, FREE_VALUE, dtype=np.uint8)))
    roi_path = _write_roi(tmp_path, shape, map_path)
    site_path = _write_site(tmp_path, shape, map_path, anchor_validated=True)

    output = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "clip_structural_endpoint_uncertainty_roi_to_site.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--roi",
            str(roi_path),
            "--site-interior",
            str(site_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        (output / "structural_endpoint_uncertainty_roi.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "site_interior_clipped_structural_endpoint_uncertainty_roi"
    assert payload["regions"]["entry_conservative_outward"]["clipped_cell_count"] == 1
    clipped = np.load(output / payload["mask_files"]["entry_conservative_outward"])
    assert int(np.count_nonzero(clipped)) == 1
    assert payload["sources"]["unbounded_roi"] == str(roi_path.resolve())
    assert payload["sources"]["site_interior"] == str(site_path.resolve())
    assert payload["site_interior_anchor_validated"] is True
    assert (output / "structural_endpoint_uncertainty_roi.png").exists()
    assert "navigation_map_modified: false" in completed.stdout


def test_site_clip_cli_rejects_topology_only_site_mask_even_if_status_says_ok(tmp_path):
    shape = (5, 8)
    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.flipud(np.full(shape, FREE_VALUE, dtype=np.uint8)))
    roi_path = _write_roi(tmp_path, shape, map_path)
    site_path = _write_site(tmp_path, shape, map_path, anchor_validated=False)
    output = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "clip_structural_endpoint_uncertainty_roi_to_site.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--roi",
            str(roi_path),
            "--site-interior",
            str(site_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "anchor-validated" in completed.stderr
