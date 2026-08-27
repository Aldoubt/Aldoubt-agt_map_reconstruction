import numpy as np
import pytest

from agt_map_reconstruction.maps.endpoint_ground_reference_confidence import (
    audit_endpoint_ground_reference_confidence,
)
from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE, FREE_VALUE


def _endpoint_envelope():
    return {
        "schema_version": 1,
        "radius_m": 0.20,
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "row_cross_span": [1.0, 4.0],
        "sides": {
            "entry": {
                "endpoint_fit": {
                    "slope_du_dv": 0.0,
                    "intercept_u": 3.0,
                }
            },
            "exit": {
                "endpoint_fit": {
                    "slope_du_dv": 0.0,
                    "intercept_u": 7.0,
                }
            },
        },
    }


def _inputs():
    base = np.full((6, 10), FREE_VALUE, dtype=np.uint8)
    base[1:5, :3] = UNKNOWN_VALUE
    base[1:5, 8:] = UNKNOWN_VALUE

    nearest = np.zeros(base.shape, dtype=np.float32)
    nearest[1:5, :3] = np.array([1.0, 2.0, 3.0])[None, :]
    nearest[1:5, 8:] = np.array([0.5, 1.5])[None, :]

    k8 = np.zeros(base.shape, dtype=np.float32)
    k16 = np.zeros(base.shape, dtype=np.float32)
    k32 = np.zeros(base.shape, dtype=np.float32)
    k16[1:5, :3] = 0.02
    k32[1:5, :3] = 0.05
    k16[1:5, 8:] = 0.01
    k32[1:5, 8:] = 0.03

    valid8 = np.ones(base.shape, dtype=bool)
    valid8[1, 0] = False
    valid = np.ones(base.shape, dtype=bool)

    models = {
        "k8": {"reference": k8, "valid_mask": valid8, "neighbor_count": 8},
        "k16": {"reference": k16, "valid_mask": valid, "neighbor_count": 16},
        "k32": {"reference": k32, "valid_mask": valid, "neighbor_count": 32},
    }
    return base, nearest, models


def test_endpoint_audit_reports_support_distance_and_cross_model_disagreement():
    base, nearest, models = _inputs()
    result = audit_endpoint_ground_reference_confidence(
        base,
        _endpoint_envelope(),
        models,
        nearest,
    )

    entry = result["sides"]["entry"]
    exit_ = result["sides"]["exit"]

    assert entry["unknown_cell_count"] == 12
    assert entry["nearest_support_distance_median_m"] == pytest.approx(2.0)
    assert entry["nearest_support_distance_p95_m"] == pytest.approx(3.0)
    assert entry["models"]["k8"]["valid_unknown_fraction"] == pytest.approx(11 / 12)
    assert entry["models"]["k16"]["valid_unknown_fraction"] == pytest.approx(1.0)
    assert entry["cross_model_disagreement"]["common_valid_unknown_cell_count"] == 11
    assert entry["cross_model_disagreement"]["range_median_m"] == pytest.approx(0.05)
    assert entry["pairwise_abs_difference"]["k8__k16"]["p95_m"] == pytest.approx(0.02)

    assert exit_["unknown_cell_count"] == 8
    assert exit_["nearest_support_distance_median_m"] == pytest.approx(1.0)
    assert exit_["cross_model_disagreement"]["range_median_m"] == pytest.approx(0.03)
    assert exit_["pairwise_abs_difference"]["k8__k16"]["p95_m"] == pytest.approx(0.01)


def test_endpoint_audit_rejects_inconsistent_nearest_support_distance_grids():
    base, nearest, models = _inputs()
    models["k16"]["nearest_support_distance_m"] = nearest + 0.1
    models["k8"]["nearest_support_distance_m"] = nearest
    models["k32"]["nearest_support_distance_m"] = nearest

    with pytest.raises(ValueError, match="nearest-support distance"):
        audit_endpoint_ground_reference_confidence(
            base,
            _endpoint_envelope(),
            models,
            nearest,
        )
