import json
import subprocess
import sys
from pathlib import Path


def _payload(repeated_count):
    def scope():
        return {
            "roi_cell_count": 10,
            "unknown_cell_count": 4,
            "classes": {
                "occupied": {"count": 2, "fraction_of_roi": 0.2},
                "known_free": {"count": 4, "fraction_of_roi": 0.4},
                "unknown_no_ground_reference": {"count": 1, "fraction_of_roi": 0.1, "fraction_of_unknown": 0.25},
                "unknown_ground_reference_no_observation": {"count": 1, "fraction_of_roi": 0.1, "fraction_of_unknown": 0.25},
                "unknown_single_scan_support": {"count": 2 - repeated_count, "fraction_of_roi": (2 - repeated_count) / 10.0, "fraction_of_unknown": (2 - repeated_count) / 4.0},
                "unknown_repeated_scan_support": {"count": repeated_count, "fraction_of_roi": repeated_count / 10.0, "fraction_of_unknown": repeated_count / 4.0},
            },
        }

    return {
        "schema_version": 1,
        "min_repeated_scans": 2,
        "full_map": scope(),
        "endpoint_rois": {"entry": scope(), "exit": scope()},
    }


def test_cli_compares_counts_without_selecting_configuration(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_payload(0)), encoding="utf-8")
    candidate.write_text(json.dumps(_payload(2)), encoding="utf-8")
    output = tmp_path / "out"
    tool = Path(__file__).resolve().parents[1] / "tools" / "compare_observation_sufficiency_layers.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads((output / "observation_sufficiency_comparison.json").read_text())
    row = next(
        item
        for item in result["rows"]
        if item["scope"] == "entry" and item["class_name"] == "unknown_repeated_scan_support"
    )
    assert row["baseline_count"] == 0
    assert row["candidate_count"] == 2
    assert row["count_delta"] == 2
    assert result["policy"]["automatic_acceptance"] is False
    assert result["policy"]["semantic_promotion"] is False
    assert "automatic_acceptance: false" in completed.stdout
