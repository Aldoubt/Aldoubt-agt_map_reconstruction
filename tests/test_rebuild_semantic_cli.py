import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.ground_evidence import EvidenceClass


def test_rebuild_semantic_cli_reuses_existing_evidence_without_pcd(tmp_path):
    evidence = np.zeros((60, 80), dtype=np.uint8)
    evidence[8:16, 5:75] = EvidenceClass.FREE_CONFIRMED
    evidence[26:34, 5:75] = EvidenceClass.FREE_CONFIRMED
    evidence[44:52, 5:75] = EvidenceClass.FREE_CONFIRMED
    evidence[18:21, 5:75] = EvidenceClass.OCCUPIED_CONFIRMED
    evidence[36:39, 5:75] = EvidenceClass.OCCUPIED_CONFIRMED
    evidence[10, 40] = EvidenceClass.OCCUPIED_CONFIRMED

    evidence_path = tmp_path / "evidence.npy"
    np.save(evidence_path, evidence)
    source_manifest = {
        "schema_version": 1,
        "grid": {
            "frame_id": "map",
            "resolution": 0.05,
            "origin": [-2.0, 3.0, 0.0],
            "width": 80,
            "height": 60,
        },
        "geometry_policy": {
            "include_interpolated": True,
            "min_longitudinal_support_ratio": 0.5,
            "min_width_m": 0.2,
            "min_length_m": 1.0,
            "promote_aisle_prior_to_static_free": False,
        },
    }
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    output = tmp_path / "rebuilt"

    script = Path(__file__).resolve().parents[1] / "tools" / "rebuild_semantic_navigation_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--evidence", str(evidence_path),
            "--manifest", str(manifest_path),
            "--output", str(output),
            "--row-direction", "1", "0",
            "--occupied-aisle-conflicts", "candidate",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    rebuilt = json.loads((output / "semantic_manifest.json").read_text())
    assert rebuilt["grid"] == source_manifest["grid"]
    assert rebuilt["row_direction"] == [1.0, 0.0]
    assert rebuilt["raw_row_band_count"] == 3
    assert rebuilt["aisle_count"] == 3
    assert rebuilt["open_area_candidate_count"] == 0
    assert rebuilt["geometry_policy"]["occupied_aisle_conflict_policy"] == "candidate"
    assert rebuilt["aisle_conflict_candidate_count"] == 1
    assert "raw_row_bands: 3" in completed.stdout
    assert "aisles: 3" in completed.stdout
    assert "open_area_candidates: 0" in completed.stdout
    assert (output / "row_band_regions.json").is_file()
    assert (output / "navigation" / "navigation_base_map.yaml").is_file()
    assert (output / "rebuild_manifest.json").is_file()
