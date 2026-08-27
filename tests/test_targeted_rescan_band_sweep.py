import numpy as np
import pytest

from agt_map_reconstruction.maps.targeted_rescan_band_sweep import (
    summarize_targeted_rescan_depth_bands,
)
from agt_map_reconstruction.maps.targeted_rescan_requirement import (
    RESCAN_KNOWN_FREE,
    RESCAN_NO_GROUND_REFERENCE,
    RESCAN_NO_OBSERVATION,
    RESCAN_REPEATED_SCAN_ANCHOR,
    RESCAN_SINGLE_SCAN_REVISIT,
)


def test_depth_band_sweep_is_cumulative_and_does_not_select_a_band():
    labels = np.array(
        [[
            RESCAN_KNOWN_FREE,
            RESCAN_REPEATED_SCAN_ANCHOR,
            RESCAN_SINGLE_SCAN_REVISIT,
            RESCAN_NO_OBSERVATION,
            RESCAN_NO_GROUND_REFERENCE,
        ]],
        dtype=np.uint8,
    )
    roi = np.ones_like(labels, dtype=bool)
    depth_cells = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=float)

    result = summarize_targeted_rescan_depth_bands(
        labels,
        roi,
        depth_cells,
        resolution_m=0.5,
        max_outward_depth_m_values=[0.5, 1.0, 2.0],
    )

    assert result["automatic_band_selection"] is False
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False

    b05, b10, b20 = result["bands"]
    assert b05["roi_cell_count"] == 2
    assert b05["rescan_required_cell_count"] == 0
    assert b05["repeated_scan_anchor_cell_count"] == 1

    assert b10["roi_cell_count"] == 3
    assert b10["rescan_required_cell_count"] == 1
    assert b10["classes"]["rescan_single_scan_revisit"]["count"] == 1

    assert b20["roi_cell_count"] == 5
    assert b20["rescan_required_cell_count"] == 3
    assert b20["classes"]["rescan_ground_known_no_observation"]["count"] == 1
    assert b20["classes"]["rescan_no_ground_reference"]["count"] == 1
    assert b20["rescan_required_fraction_of_endpoint_roi"] == pytest.approx(3.0 / 5.0)


def test_depth_band_sweep_rejects_invalid_inputs():
    labels = np.full((2, 2), RESCAN_NO_GROUND_REFERENCE, dtype=np.uint8)
    roi = np.ones((2, 2), dtype=bool)
    depth = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="resolution_m"):
        summarize_targeted_rescan_depth_bands(
            labels,
            roi,
            depth,
            resolution_m=0.0,
            max_outward_depth_m_values=[1.0],
        )

    with pytest.raises(ValueError, match="unique"):
        summarize_targeted_rescan_depth_bands(
            labels,
            roi,
            depth,
            resolution_m=0.05,
            max_outward_depth_m_values=[1.0, 1.0],
        )
