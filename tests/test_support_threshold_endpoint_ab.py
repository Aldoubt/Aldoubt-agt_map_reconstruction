import numpy as np
import pytest

from agt_map_reconstruction.maps.support_threshold_endpoint_ab import (
    evaluate_support_thresholds,
)


def _baseline():
    return {
        "radius_m": 0.2,
        "eligible_row_labels": ["A01"],
        "row_axis_direction": [1.0, 0.0],
        "row_cross_span": [0.0, 1.0],
        "sides": {
            "entry": {
                "strict": {
                    "best_component": {
                        "cross_row_coverage_fraction": 0.0,
                        "endpoint_distance_median_m": 1.0,
                        "max_outward_depth_m": 0.0,
                    }
                }
            },
            "exit": {
                "strict": {
                    "best_component": {
                        "cross_row_coverage_fraction": 0.0,
                        "endpoint_distance_median_m": 1.0,
                        "max_outward_depth_m": 0.0,
                    }
                }
            },
        },
    }


def test_rejects_invalid_support_basis_before_geometry_use():
    base = np.full((4, 4), 205, dtype=np.uint8)
    counts = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="support_basis"):
        evaluate_support_thresholds(
            base,
            counts,
            [],
            [],
            resolution=0.05,
            radius_m=0.2,
            baseline_envelope=_baseline(),
            min_support_values=[1],
            support_basis="points",
        )


def test_rejects_unsorted_or_duplicate_thresholds_before_geometry_use():
    base = np.full((4, 4), 205, dtype=np.uint8)
    counts = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="unique and increasing"):
        evaluate_support_thresholds(
            base,
            counts,
            [],
            [],
            resolution=0.05,
            radius_m=0.2,
            baseline_envelope=_baseline(),
            min_support_values=[2, 1],
            support_basis="scan",
        )
    with pytest.raises(ValueError, match="unique and increasing"):
        evaluate_support_thresholds(
            base,
            counts,
            [],
            [],
            resolution=0.05,
            radius_m=0.2,
            baseline_envelope=_baseline(),
            min_support_values=[1, 1],
            support_basis="scan",
        )


def test_rejects_shape_mismatch_before_geometry_use():
    with pytest.raises(ValueError, match="shape"):
        evaluate_support_thresholds(
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((3, 4), dtype=np.uint16),
            [],
            [],
            resolution=0.05,
            radius_m=0.2,
            baseline_envelope=_baseline(),
            min_support_values=[1],
            support_basis="ray",
        )
