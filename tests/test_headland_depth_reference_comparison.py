import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.headland_depth_reference_comparison import (
    compare_headland_depth_with_unbounded,
)


def _finite_evidence():
    return {
        "method": "finite_headland_depth_observation_sufficiency",
        "entry": {
            "bands": [
                {
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                    "unknown_cell_count": 10,
                    "trusted_ground_unknown_cell_count": 8,
                    "ground_reference_ceiling_fraction_of_unknown": 0.8,
                },
                {
                    "depth_min_m": 0.5,
                    "depth_max_m": 1.0,
                    "unknown_cell_count": 30,
                    "trusted_ground_unknown_cell_count": 6,
                    "ground_reference_ceiling_fraction_of_unknown": 0.2,
                },
            ]
        },
        "exit": {
            "bands": [
                {
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                    "unknown_cell_count": 20,
                    "trusted_ground_unknown_cell_count": 10,
                    "ground_reference_ceiling_fraction_of_unknown": 0.5,
                }
            ]
        },
    }


def _unbounded_evidence():
    return {
        "method": "fused_structural_roi_observation_sufficiency",
        "entry": {
            "conservative_outward": {
                "unknown_cell_count": 100,
                "trusted_ground_unknown_cell_count": 10,
                "ground_reference_ceiling_fraction_of_unknown": 0.1,
            }
        },
        "exit": {
            "conservative_outward": {
                "unknown_cell_count": 200,
                "trusted_ground_unknown_cell_count": 10,
                "ground_reference_ceiling_fraction_of_unknown": 0.05,
            }
        },
    }


def test_comparison_keeps_spatial_domains_non_equivalent_and_aggregates_by_cells():
    result = compare_headland_depth_with_unbounded(
        _finite_evidence(),
        _unbounded_evidence(),
    )

    assert result["spatial_domains_equivalent"] is False
    assert result["historical_unbounded_metrics_used_for_acceptance"] is False
    assert result["finite_depth_profile_is_primary"] is True

    entry = result["entry"]
    assert entry["historical_unbounded"]["unknown_cell_count"] == 100
    assert entry["finite_depth_aggregate"]["unknown_cell_count"] == 40
    assert entry["finite_depth_aggregate"]["trusted_ground_unknown_cell_count"] == 14
    assert np.isclose(
        entry["finite_depth_aggregate"]["ground_reference_ceiling_fraction_of_unknown"],
        14 / 40,
    )
    # The finite curve is retained verbatim; the aggregate is not a mean of 0.8 and 0.2.
    assert entry["finite_depth_bands"] == _finite_evidence()["entry"]["bands"]
    assert result["policy"]["fraction_difference_reported_as_improvement"] is False
    assert "improvement_fraction" not in result


def test_reference_comparison_cli_writes_reference_only_json(tmp_path):
    finite_path = tmp_path / "finite.json"
    unbounded_path = tmp_path / "unbounded.json"
    finite_path.write_text(json.dumps(_finite_evidence()), encoding="utf-8")
    unbounded_path.write_text(json.dumps(_unbounded_evidence()), encoding="utf-8")
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "compare_headland_depth_with_unbounded.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--headland-depth-evidence",
            str(finite_path),
            "--unbounded-evidence",
            str(unbounded_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (output / "headland_depth_vs_unbounded_reference.json").read_text(encoding="utf-8")
    )
    assert result["spatial_domains_equivalent"] is False
    assert result["historical_unbounded_metrics_used_for_acceptance"] is False
    assert result["finite_depth_profile_is_primary"] is True
    assert result["sources"]["headland_depth_evidence"] == str(finite_path.resolve())
    assert result["sources"]["unbounded_evidence"] == str(unbounded_path.resolve())
    assert "spatial_domains_equivalent: false" in completed.stdout
