import numpy as np

from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.vehicle_envelope import (
    VehicleEnvelopeConfig,
    build_vehicle_free_mask,
    build_vehicle_navigation_layers,
)


def test_vehicle_envelope_rejects_a_corridor_narrower_than_the_body():
    evidence = np.full((25, 40), EvidenceClass.FREE_CONFIRMED, dtype=np.uint8)
    evidence[10:16, 20] = EvidenceClass.OCCUPIED_CONFIRMED
    config = VehicleEnvelopeConfig(
        length_m=0.8, width_m=0.6, resolution=0.1, min_aisle_length_m=1.0
    )
    safe = build_vehicle_free_mask(evidence, config)
    assert not safe[12, 20]
    assert safe[12, 5]


def test_vehicle_envelope_keeps_a_long_safe_aisle():
    evidence = np.full((30, 50), EvidenceClass.UNKNOWN, dtype=np.uint8)
    evidence[10:20, 3:47] = EvidenceClass.FREE_CONFIRMED
    layers = build_vehicle_navigation_layers(
        evidence,
        VehicleEnvelopeConfig(
            length_m=0.8, width_m=0.6, resolution=0.1, min_aisle_length_m=2.0
        ),
    )
    assert layers["vehicle_free"].sum() > 0
    assert layers["aisle_candidate"].sum() > 0
