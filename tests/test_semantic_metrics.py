import json

import numpy as np

from agt_map_reconstruction.maps.semantic_metrics import compute_ridge_metrics, write_ridge_metrics


def _ridge(center_x, width=1.0, length=10.0):
    # Long axis is Y; x is the transverse direction used for spacing.
    return {
        "metric_polygon_xy": [
            [center_x - width / 2, -length / 2],
            [center_x + width / 2, -length / 2],
            [center_x + width / 2, length / 2],
            [center_x - width / 2, length / 2],
        ],
        "width_m": width,
        "length_m": length,
    }


def test_ridges_are_ordered_and_include_adjacent_spacing_and_clear_gap():
    metrics = compute_ridge_metrics({
        "ridge_rectangles": [_ridge(0), _ridge(3), _ridge(6)],
    })

    assert [row["label"] for row in metrics["ridges"]] == ["R01", "R02", "R03"]
    assert [row["center_x_m"] for row in metrics["ridges"]] == [0.0, 3.0, 6.0]
    assert metrics["ridges"][1]["previous_center_spacing_m"] == 3.0
    assert metrics["ridges"][1]["next_center_spacing_m"] == 3.0
    assert metrics["ridges"][1]["previous_clear_gap_m"] == 2.0
    assert metrics["summary"]["spacing_outlier_count"] == 0


def test_width_and_spacing_outliers_are_flagged():
    metrics = compute_ridge_metrics({
        "ridge_rectangles": [_ridge(0), _ridge(2, width=3), _ridge(8)],
    })

    assert metrics["ridges"][1]["width_outlier"] is True
    assert any(row["spacing_outlier"] for row in metrics["ridges"])


def test_metrics_writer_creates_csv_and_json(tmp_path):
    write_ridge_metrics({"ridge_rectangles": [_ridge(1)]}, tmp_path)
    assert (tmp_path / "ridge_metrics.csv").read_text(encoding="utf-8").splitlines()[0].startswith("ridge_id,label")
    document = json.loads((tmp_path / "ridge_metrics.json").read_text(encoding="utf-8"))
    assert document["ridges"][0]["label"] == "R01"
