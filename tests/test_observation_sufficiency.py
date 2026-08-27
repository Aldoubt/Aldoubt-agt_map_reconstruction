import numpy as np
import pytest

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.observation_sufficiency import (
    LABEL_KNOWN_FREE,
    LABEL_OCCUPIED,
    LABEL_UNKNOWN_NO_GROUND_REFERENCE,
    LABEL_UNKNOWN_NO_OBSERVATION,
    LABEL_UNKNOWN_REPEATED_SCAN,
    LABEL_UNKNOWN_SINGLE_SCAN,
    build_observation_sufficiency_labels,
    summarize_observation_sufficiency,
)


def test_classifies_unknown_by_ground_and_unique_scan_support():
    base = np.array(
        [
            [OCCUPIED_VALUE, FREE_VALUE, UNKNOWN_VALUE],
            [UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE],
        ],
        dtype=np.uint8,
    )
    ground = np.array(
        [
            [0.0, 0.0, np.nan],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    scans = np.array(
        [
            [9, 9, 7],
            [0, 1, 3],
        ],
        dtype=np.uint16,
    )

    labels = build_observation_sufficiency_labels(
        base, ground, scans, min_repeated_scans=2
    )

    assert labels.tolist() == [
        [int(LABEL_OCCUPIED), int(LABEL_KNOWN_FREE), int(LABEL_UNKNOWN_NO_GROUND_REFERENCE)],
        [int(LABEL_UNKNOWN_NO_OBSERVATION), int(LABEL_UNKNOWN_SINGLE_SCAN), int(LABEL_UNKNOWN_REPEATED_SCAN)],
    ]


def test_summary_reports_unknown_fractions_without_semantic_promotion():
    labels = np.array(
        [
            [LABEL_OCCUPIED, LABEL_KNOWN_FREE, LABEL_UNKNOWN_NO_GROUND_REFERENCE],
            [LABEL_UNKNOWN_NO_OBSERVATION, LABEL_UNKNOWN_SINGLE_SCAN, LABEL_UNKNOWN_REPEATED_SCAN],
        ],
        dtype=np.uint8,
    )
    result = summarize_observation_sufficiency(labels)

    assert result["roi_cell_count"] == 6
    assert result["unknown_cell_count"] == 4
    assert result["classes"]["unknown_repeated_scan_support"]["count"] == 1
    assert result["classes"]["unknown_repeated_scan_support"]["fraction_of_unknown"] == pytest.approx(0.25)
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False


def test_rejects_invalid_threshold_and_shape_mismatch():
    base = np.full((2, 2), UNKNOWN_VALUE, dtype=np.uint8)
    ground = np.zeros((2, 2), dtype=np.float64)
    scans = np.zeros((2, 2), dtype=np.uint16)

    with pytest.raises(ValueError, match="min_repeated_scans"):
        build_observation_sufficiency_labels(base, ground, scans, min_repeated_scans=1)

    with pytest.raises(ValueError, match="shape mismatch"):
        build_observation_sufficiency_labels(base, ground[:1], scans)
