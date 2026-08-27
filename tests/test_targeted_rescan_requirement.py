import numpy as np

from agt_map_reconstruction.maps.observation_sufficiency import (
    LABEL_KNOWN_FREE,
    LABEL_OCCUPIED,
    LABEL_UNKNOWN_NO_GROUND_REFERENCE,
    LABEL_UNKNOWN_NO_OBSERVATION,
    LABEL_UNKNOWN_REPEATED_SCAN,
    LABEL_UNKNOWN_SINGLE_SCAN,
)
from agt_map_reconstruction.maps.targeted_rescan_requirement import (
    RESCAN_KNOWN_FREE,
    RESCAN_NO_GROUND_REFERENCE,
    RESCAN_NO_OBSERVATION,
    RESCAN_OCCUPIED,
    RESCAN_OUTSIDE_ENDPOINT,
    RESCAN_REPEATED_SCAN_ANCHOR,
    RESCAN_SINGLE_SCAN_REVISIT,
    build_targeted_rescan_requirement,
    summarize_targeted_rescan_requirement,
)


def test_build_targeted_rescan_requirement_preserves_evidence_semantics():
    suff = np.array(
        [
            [LABEL_OCCUPIED, LABEL_KNOWN_FREE, LABEL_UNKNOWN_NO_GROUND_REFERENCE],
            [LABEL_UNKNOWN_NO_OBSERVATION, LABEL_UNKNOWN_SINGLE_SCAN, LABEL_UNKNOWN_REPEATED_SCAN],
        ],
        dtype=np.uint8,
    )
    roi = np.array([[True, True, True], [True, True, False]])

    result = build_targeted_rescan_requirement(suff, roi)

    assert result[0, 0] == RESCAN_OCCUPIED
    assert result[0, 1] == RESCAN_KNOWN_FREE
    assert result[0, 2] == RESCAN_NO_GROUND_REFERENCE
    assert result[1, 0] == RESCAN_NO_OBSERVATION
    assert result[1, 1] == RESCAN_SINGLE_SCAN_REVISIT
    assert result[1, 2] == RESCAN_OUTSIDE_ENDPOINT


def test_summary_separates_rescan_required_from_repeated_scan_anchor():
    labels = np.array(
        [
            [RESCAN_NO_GROUND_REFERENCE, RESCAN_NO_OBSERVATION, RESCAN_SINGLE_SCAN_REVISIT],
            [RESCAN_REPEATED_SCAN_ANCHOR, RESCAN_KNOWN_FREE, RESCAN_OCCUPIED],
        ],
        dtype=np.uint8,
    )
    roi = np.ones(labels.shape, dtype=bool)

    summary = summarize_targeted_rescan_requirement(labels, roi_mask=roi)

    assert summary["roi_cell_count"] == 6
    assert summary["rescan_required_cell_count"] == 3
    assert summary["repeated_scan_anchor_cell_count"] == 1
    assert summary["rescan_required_components"]["component_count"] == 1
    assert summary["navigation_map_modified"] is False
    assert summary["semantic_promotion"] is False
