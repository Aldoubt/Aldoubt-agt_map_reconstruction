import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_ground_reference_consensus_cli_writes_confidence_assets(tmp_path):
    a_dir = tmp_path / "k8"
    b_dir = tmp_path / "k16"
    a_dir.mkdir()
    b_dir.mkdir()

    a = np.array([[0.0, 0.1, 0.2]], dtype=np.float32)
    b = np.array([[0.02, 0.12, 0.5]], dtype=np.float32)
    distance = np.array([[0.1, 1.0, 0.2]], dtype=np.float32)
    np.save(a_dir / "ground_reference.npy", a)
    np.save(b_dir / "ground_reference.npy", b)
    np.save(a_dir / "ground_reference_nearest_support_distance.npy", distance)
    np.save(b_dir / "ground_reference_nearest_support_distance.npy", distance)

    for directory, k in ((a_dir, 8), (b_dir, 16)):
        (directory / "ground_reference_manifest.json").write_text(
            json.dumps({
                "model": {"neighbor_count": k},
                "policy": {"semantic_promotion": False},
            }),
            encoding="utf-8",
        )

    output = tmp_path / "consensus"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_ground_reference_consensus.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--reference-a", str(a_dir),
        "--reference-b", str(b_dir),
        "--output", str(output),
        "--max-support-distance-m", "2.0",
        "--max-model-disagreement-m", "0.05",
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    reference = np.load(output / "ground_reference.npy")
    mask = np.load(output / "ground_reference_confidence_mask.npy")
    disagreement = np.load(output / "ground_reference_model_disagreement.npy")
    manifest = json.loads(
        (output / "ground_reference_consensus_manifest.json").read_text(encoding="utf-8")
    )

    np.testing.assert_allclose(reference[0, :2], [0.01, 0.11])
    assert np.isnan(reference[0, 2])
    np.testing.assert_array_equal(mask, [[1, 1, 0]])
    assert disagreement[0, 2] == np.float32(0.3)
    assert manifest["summary"]["accepted_cell_count"] == 2
    assert manifest["policy"]["semantic_promotion"] is False
    assert "semantic_promotion: false" in completed.stdout
