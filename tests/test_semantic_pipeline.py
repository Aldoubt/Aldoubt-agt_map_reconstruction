from types import SimpleNamespace

import numpy as np
import pytest

from agt_map_reconstruction.maps.semantic_pipeline import (
    infer_row_direction_from_evidence,
    metadata_from_statistics,
)
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass


def test_infer_row_direction_from_horizontal_evidence_bands():
    evidence = np.zeros((20, 80), dtype=np.uint8)
    evidence[4:7, 5:75] = EvidenceClass.FREE_CONFIRMED
    evidence[13:16, 5:75] = EvidenceClass.FREE_CONFIRMED

    direction = infer_row_direction_from_evidence(evidence)

    assert abs(direction[0]) > 0.99
    assert abs(direction[1]) < 0.05


def test_infer_row_direction_rejects_insufficient_support():
    evidence = np.zeros((5, 5), dtype=np.uint8)
    evidence[2, 2] = EvidenceClass.FREE_CONFIRMED

    with pytest.raises(ValueError, match="row direction"):
        infer_row_direction_from_evidence(evidence)


def test_metadata_from_statistics_preserves_origin_resolution_and_shape():
    statistics = SimpleNamespace(
        low_height=np.zeros((12, 34), dtype=float),
        origin_xy=np.array([-2.5, 7.25]),
        resolution=0.05,
    )

    metadata = metadata_from_statistics(statistics)

    assert metadata.origin_x == -2.5
    assert metadata.origin_y == 7.25
    assert metadata.resolution == 0.05
    assert metadata.width == 34
    assert metadata.height == 12
    assert metadata.frame_id == "map"
