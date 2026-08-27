import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.ray_endpoint_evidence_ab import (
    build_observation_evaluation_overlay,
    compare_endpoint_envelopes,
)


def test_overlay_only_promotes_ray_supported_unknown_cells():
    base = np.array(
        [
            [FREE_VALUE, UNKNOWN_VALUE, OCCUPIED_VALUE],
            [UNKNOWN_VALUE, UNKNOWN_VALUE, FREE_VALUE],
        ],
        dtype=np.uint8,
    )
    support = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    )

    overlay, summary = build_observation_evaluation_overlay(base, support)

    assert overlay[0, 0] == FREE_VALUE
    assert overlay[0, 1] == FREE_VALUE
    assert overlay[0, 2] == OCCUPIED_VALUE
    assert overlay[1, 0] == UNKNOWN_VALUE
    assert overlay[1, 1] == FREE_VALUE
    assert overlay[1, 2] == FREE_VALUE
    assert summary["ray_supported_unknown_cell_count"] == 2
    assert summary["ray_supported_occupied_cell_count_ignored"] == 1
    assert summary["navigation_map_modified"] is False
    assert summary["semantic_promotion"] is False


def _envelope(coverage, endpoint, depth):
    side = {
        "strict": {
            "best_component": {
                "cross_row_coverage_fraction": coverage,
                "endpoint_distance_median_m": endpoint,
                "max_outward_depth_m": depth,
            }
        }
    }
    return {
        "radius_m": 0.2,
        "eligible_row_labels": ["A01", "A02"],
        "row_axis_direction": [1.0, 0.0],
        "row_cross_span": [0.0, 10.0],
        "sides": {"entry": side, "exit": side},
    }


def test_comparison_reports_improvement_with_frozen_geometry():
    baseline = _envelope(0.1, 4.0, 0.2)
    candidate = _envelope(0.4, 1.5, 1.1)

    result = compare_endpoint_envelopes(baseline, candidate)

    entry = result["sides"]["entry"]
    assert entry["delta"]["cross_row_coverage_fraction"] == 0.30000000000000004
    assert entry["delta"]["endpoint_distance_reduction_m"] == 2.5
    assert entry["delta"]["max_outward_depth_gain_m"] == 0.9000000000000001
    assert result["geometry_frozen"] is True
    assert result["automatic_acceptance"] is False
    assert result["semantic_promotion"] is False
