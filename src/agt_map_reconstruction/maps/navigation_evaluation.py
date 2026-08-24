"""Common metrics for segmentation outputs and MK-mini envelope previews."""

import numpy as np

from .ground_evidence import EvidenceClass
from .vehicle_envelope import VehicleEnvelopeConfig, build_vehicle_navigation_layers


def pmf_grids_to_evidence(ground_grid, observed_grid):
    ground_grid = np.asarray(ground_grid, dtype=bool)
    observed_grid = np.asarray(observed_grid, dtype=bool)
    if ground_grid.shape != observed_grid.shape:
        raise ValueError("ground_grid and observed_grid must have matching shapes")
    evidence = np.full(ground_grid.shape, EvidenceClass.UNKNOWN, dtype=np.uint8)
    evidence[observed_grid & ~ground_grid] = EvidenceClass.OCCUPIED_CONFIRMED
    evidence[ground_grid] = EvidenceClass.FREE_CONFIRMED
    return evidence


def evaluate_navigation_method(name, evidence, envelope):
    """Return comparable evidence and body-envelope metrics."""
    layers = build_vehicle_navigation_layers(evidence, envelope)
    values = np.asarray(evidence)
    measured = values != EvidenceClass.UNKNOWN
    return {
        "method": name,
        "grid_shape_yx": list(values.shape),
        "measured_cells": int(measured.sum()),
        "free_cells": int((values == EvidenceClass.FREE_CONFIRMED).sum()),
        "occupied_cells": int((values == EvidenceClass.OCCUPIED_CONFIRMED).sum()),
        "unknown_cells": int((values == EvidenceClass.UNKNOWN).sum()),
        "vehicle_safe_cells": int(layers["vehicle_free"].sum()),
        "aisle_candidate_cells": int(layers["aisle_candidate"].sum()),
    }, layers
