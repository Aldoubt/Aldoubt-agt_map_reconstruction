import numpy as np

from agt_map_reconstruction.maps.endpoint_ground_reference_gate_sweep import (
    sweep_endpoint_ground_reference_gate,
)


def _envelope():
    return {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "row_cross_span": [1.0, 4.0],
        "sides": {
            "entry": {"endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 3.0}},
            "exit": {"endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 7.0}},
        },
    }


def test_gate_sweep_reports_joint_unknown_acceptance():
    unknown = np.zeros((6, 10), dtype=bool)
    unknown[1:5, :3] = True
    unknown[1:5, 8:] = True

    distance = np.zeros((6, 10), dtype=float)
    distance[1:5, :3] = np.array([0.5, 1.0, 2.0])[None, :]
    distance[1:5, 8:] = np.array([0.25, 1.5])[None, :]

    disagreement = np.zeros((6, 10), dtype=float)
    disagreement[1:5, :3] = np.array([0.02, 0.08, 0.20])[None, :]
    disagreement[1:5, 8:] = np.array([0.03, 0.12])[None, :]

    result = sweep_endpoint_ground_reference_gate(
        unknown,
        _envelope(),
        distance,
        disagreement,
        max_support_distances_m=[0.75, 2.0],
        max_model_disagreements_m=[0.05, 0.15],
    )

    entry = result["sides"]["entry"]
    exit_ = result["sides"]["exit"]
    assert entry["unknown_cell_count"] == 12
    assert exit_["unknown_cell_count"] == 8

    # Entry: only the first of three columns passes the strictest gate.
    assert entry["grid"][0]["accepted_unknown_fraction"] == 1 / 3
    # Entry: first two columns pass at distance<=2.0 and disagreement<=0.15.
    assert entry["grid"][-1]["accepted_unknown_fraction"] == 2 / 3
    # Exit: first column passes strictest gate; both pass at relaxed gate.
    assert exit_["grid"][0]["accepted_unknown_fraction"] == 1 / 2
    assert exit_["grid"][-1]["accepted_unknown_fraction"] == 1.0
    assert result["semantic_promotion"] is False
