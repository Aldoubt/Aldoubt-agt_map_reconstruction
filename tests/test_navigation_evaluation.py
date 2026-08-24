import numpy as np

from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.navigation_evaluation import (
    evaluate_navigation_method,
    pmf_grids_to_evidence,
)
from agt_map_reconstruction.maps.vehicle_envelope import VehicleEnvelopeConfig


def test_pmf_grids_preserve_unknown_and_obstacle_states():
    evidence = pmf_grids_to_evidence(
        np.array([[True, False]]), np.array([[True, True]])
    )
    np.testing.assert_array_equal(
        evidence, [[EvidenceClass.FREE_CONFIRMED, EvidenceClass.OCCUPIED_CONFIRMED]]
    )


def test_evaluation_reports_vehicle_safe_area():
    evidence = np.full((20, 40), EvidenceClass.FREE_CONFIRMED, dtype=np.uint8)
    metrics, layers = evaluate_navigation_method(
        "test", evidence, VehicleEnvelopeConfig(resolution=0.1)
    )
    assert metrics["vehicle_safe_cells"] == 800
    assert layers["aisle_candidate"].any()
