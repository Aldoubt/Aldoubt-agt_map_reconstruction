import numpy as np
import pytest

from agt_map_reconstruction.maps.ground_reference_consensus import (
    build_ground_reference_consensus,
)


def test_consensus_keeps_only_nearby_agreeing_reference_cells():
    k8 = np.array([[0.00, 0.10, 0.20, 0.30]], dtype=float)
    k16 = np.array([[0.02, 0.12, 0.50, 0.31]], dtype=float)
    distance = np.array([[0.10, 1.50, 0.20, 4.00]], dtype=float)

    result = build_ground_reference_consensus(
        k8,
        k16,
        distance,
        max_support_distance_m=2.0,
        max_model_disagreement_m=0.05,
    )

    np.testing.assert_allclose(
        result["ground_reference"][0, :2],
        [0.01, 0.11],
    )
    assert np.isnan(result["ground_reference"][0, 2])
    assert np.isnan(result["ground_reference"][0, 3])
    np.testing.assert_array_equal(result["confidence_mask"], [[True, True, False, False]])
    assert result["summary"]["accepted_cell_count"] == 2
    assert result["summary"]["rejected_disagreement_cell_count"] == 1
    assert result["summary"]["rejected_distance_cell_count"] == 1
    assert result["summary"]["semantic_promotion"] is False


def test_consensus_thresholds_must_be_explicit_positive_values():
    reference = np.zeros((2, 2), dtype=float)
    distance = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="max_support_distance_m"):
        build_ground_reference_consensus(
            reference,
            reference,
            distance,
            max_support_distance_m=0.0,
            max_model_disagreement_m=0.1,
        )

    with pytest.raises(ValueError, match="max_model_disagreement_m"):
        build_ground_reference_consensus(
            reference,
            reference,
            distance,
            max_support_distance_m=1.0,
            max_model_disagreement_m=0.0,
        )
