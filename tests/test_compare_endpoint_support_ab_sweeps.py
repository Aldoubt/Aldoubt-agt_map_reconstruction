import json

import pytest

from tools.compare_endpoint_support_ab_sweeps import load_rows


def _payload(basis="scan", gain=0.1):
    return {
        "support_basis": basis,
        "thresholds": [
            {
                "min_support": 2,
                "supported_cell_count": 12,
                "overlay_summary": {"ray_supported_unknown_cell_count": 4},
                "comparison": {
                    "geometry_frozen": True,
                    "sides": {
                        "entry": {
                            "baseline": {
                                "cross_row_coverage_fraction": 0.1,
                                "endpoint_distance_median_m": 2.0,
                                "max_outward_depth_m": 0.2,
                            },
                            "candidate": {
                                "cross_row_coverage_fraction": 0.1 + gain,
                                "endpoint_distance_median_m": 1.5,
                                "max_outward_depth_m": 0.3,
                            },
                            "delta": {
                                "cross_row_coverage_fraction": gain,
                                "endpoint_distance_reduction_m": 0.5,
                                "max_outward_depth_gain_m": 0.1,
                            },
                        },
                        "exit": {
                            "baseline": {
                                "cross_row_coverage_fraction": 0.2,
                                "endpoint_distance_median_m": 3.0,
                                "max_outward_depth_m": 0.4,
                            },
                            "candidate": {
                                "cross_row_coverage_fraction": 0.2,
                                "endpoint_distance_median_m": 3.0,
                                "max_outward_depth_m": 0.4,
                            },
                            "delta": {
                                "cross_row_coverage_fraction": 0.0,
                                "endpoint_distance_reduction_m": 0.0,
                                "max_outward_depth_gain_m": 0.0,
                            },
                        },
                    },
                },
            }
        ],
    }


def test_load_rows_flattens_measured_d3_deltas(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(_payload(gain=0.1)), encoding="utf-8")
    second.write_text(json.dumps(_payload(gain=0.2)), encoding="utf-8")

    result = load_rows([f"a={first}", f"b={second}"])

    assert result["support_basis"] == "scan"
    assert len(result["rows"]) == 4
    row = next(
        item
        for item in result["rows"]
        if item["label"] == "b" and item["side"] == "entry"
    )
    assert row["min_support"] == 2
    assert row["coverage_gain"] == pytest.approx(0.2)
    assert row["endpoint_reduction_m"] == pytest.approx(0.5)
    assert result["policy"]["automatic_threshold_selection"] is False


def test_load_rows_rejects_mixed_support_basis(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(_payload("scan")), encoding="utf-8")
    second.write_text(json.dumps(_payload("ray")), encoding="utf-8")

    with pytest.raises(ValueError, match="same support_basis"):
        load_rows([f"a={first}", f"b={second}"])
