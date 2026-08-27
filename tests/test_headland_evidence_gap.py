from agt_map_reconstruction.maps.headland_evidence_gap import (
    attach_endpoint_evidence_gaps,
    summarize_endpoint_evidence_gap,
)


def _component(coverage, endpoint, depth, unknown):
    return {
        "cross_row_coverage_fraction": coverage,
        "endpoint_distance_median_m": endpoint,
        "max_outward_depth_m": depth,
        "unknown_cell_fraction": unknown,
    }


def test_summarizes_unknown_dominated_relaxed_gain_without_promotion():
    side = {
        "strict": {"best_component": _component(0.02, 7.5, 0.2, 0.0)},
        "relaxed_unknown_allowed": {
            "best_component": _component(1.0, 0.4, 10.2, 0.998)
        },
    }

    gap = summarize_endpoint_evidence_gap(side)

    assert gap["available"] is True
    assert abs(gap["coverage_gain"] - 0.98) < 1e-12
    assert abs(gap["endpoint_distance_reduction_m"] - 7.1) < 1e-12
    assert abs(gap["outward_depth_gain_m"] - 10.0) < 1e-12
    assert abs(gap["relaxed_observed_fraction"] - 0.002) < 1e-12
    assert gap["semantic_promotion"] is False


def test_attach_adds_both_side_summaries_and_keeps_diagnostic_policy():
    side = {
        "strict": {"best_component": _component(0.10, 1.0, 0.5, 0.0)},
        "relaxed_unknown_allowed": {
            "best_component": _component(0.90, 0.2, 2.0, 0.75)
        },
    }
    result = {"sides": {"entry": dict(side), "exit": dict(side)}}

    enriched = attach_endpoint_evidence_gaps(result)

    assert enriched["sides"]["entry"]["evidence_gap"]["coverage_gain"] == 0.8
    assert enriched["sides"]["exit"]["evidence_gap"]["relaxed_observed_fraction"] == 0.25
    assert enriched["evidence_gap_policy"]["semantic_promotion"] is False
