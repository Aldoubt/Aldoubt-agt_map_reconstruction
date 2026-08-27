import numpy as np

from agt_map_reconstruction.maps.row_band_classification import (
    classify_row_bands,
    robust_upper_width_threshold,
)


def _band(index, width_m):
    return {
        "aisle_id": index,
        "label": f"A{index:02d}",
        "polygon_xy": [[0.0, 0.0], [100.0, 0.0], [100.0, 1.0], [0.0, 1.0]],
        "centerline_xy": [[0.0, 0.5], [100.0, 0.5]],
        "width_m": float(width_m),
        "length_m": 30.0,
        "heading_rad": 0.0,
    }


def test_realistic_width_distribution_separates_only_three_wide_bands():
    widths = [
        0.35, 0.70, 0.70, 0.55, 0.65, 0.40, 0.35, 0.75, 0.60, 0.70,
        0.80, 0.50, 0.75, 0.95, 0.80, 1.00, 0.50, 1.75, 2.90, 4.65,
    ]
    bands = [_band(index, width) for index, width in enumerate(widths, start=1)]

    result = classify_row_bands(bands)

    assert np.isclose(robust_upper_width_threshold(widths), 1.2875)
    assert np.isclose(result.width_outlier_threshold_m, 1.2875)
    assert [item["source_band_label"] for item in result.row_aisles] == [
        f"A{index:02d}" for index in range(1, 18)
    ]
    assert [item["label"] for item in result.row_aisles] == [
        f"A{index:02d}" for index in range(1, 18)
    ]
    assert [item["source_band_label"] for item in result.open_area_candidates] == [
        "A18", "A19", "A20"
    ]
    assert [item["label"] for item in result.open_area_candidates] == [
        "O01", "O02", "O03"
    ]
    assert all(
        item["region_class"] == "wide_open_area_candidate"
        for item in result.open_area_candidates
    )


def test_small_band_set_is_not_forced_into_outlier_classification():
    bands = [_band(1, 0.50), _band(2, 2.00), _band(3, 4.00)]

    result = classify_row_bands(bands)

    assert result.width_outlier_threshold_m is None
    assert len(result.row_aisles) == 3
    assert result.open_area_candidates == []
