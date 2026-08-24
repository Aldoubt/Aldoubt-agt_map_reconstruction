"""Vehicle-specific, row-aligned navigation layers for a fixed PCD map."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import ndimage

from .ground_evidence import EvidenceClass


@dataclass(frozen=True)
class VehicleEnvelopeConfig:
    """Offline MK-mini body envelope, expressed in the row-aligned map frame."""

    length_m: float = 0.840
    width_m: float = 0.600
    resolution: float = 0.05
    safety_margin_m: float = 0.0
    min_aisle_length_m: float = 1.0

    def __post_init__(self):
        for name in ("length_m", "width_m", "resolution", "min_aisle_length_m"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.safety_margin_m, Real) or not np.isfinite(self.safety_margin_m) or self.safety_margin_m < 0:
            raise ValueError("safety_margin_m must be finite and non-negative")


def _rectangle_structure(config):
    length = int(np.ceil((config.length_m + 2 * config.safety_margin_m) / config.resolution))
    width = int(np.ceil((config.width_m + 2 * config.safety_margin_m) / config.resolution))
    return np.ones((max(1, width), max(1, length)), dtype=bool)


def build_vehicle_free_mask(evidence, config):
    """Return cell centers where the complete MK-mini rectangle has support."""
    if not isinstance(config, VehicleEnvelopeConfig):
        raise TypeError("config must be a VehicleEnvelopeConfig")
    evidence = np.asarray(evidence)
    if evidence.ndim != 2:
        raise ValueError("evidence must be two-dimensional")
    free = evidence == EvidenceClass.FREE_CONFIRMED
    occupied = evidence == EvidenceClass.OCCUPIED_CONFIRMED
    # The map is row-aligned: length is along columns and width along rows.
    blocked_centers = ndimage.binary_dilation(occupied, structure=_rectangle_structure(config))
    return free & ~blocked_centers


def extract_row_aisles(vehicle_free, config):
    """Keep long, connected vehicle-safe regions and reject isolated pixels."""
    if not isinstance(config, VehicleEnvelopeConfig):
        raise TypeError("config must be a VehicleEnvelopeConfig")
    vehicle_free = np.asarray(vehicle_free, dtype=bool)
    minimum_length = max(1, int(np.ceil(config.min_aisle_length_m / config.resolution)))
    labels, count = ndimage.label(vehicle_free, structure=np.ones((3, 3), dtype=bool))
    result = np.zeros_like(vehicle_free)
    for label in range(1, count + 1):
        rows, columns = np.where(labels == label)
        if len(columns) == 0 or columns.max() - columns.min() + 1 < minimum_length:
            continue
        result[labels == label] = True
    return result


def build_vehicle_navigation_layers(evidence, config):
    """Build measured free, vehicle-safe free, and aisle candidate layers."""
    vehicle_free = build_vehicle_free_mask(evidence, config)
    return {
        "measured_free": np.asarray(evidence) == EvidenceClass.FREE_CONFIRMED,
        "vehicle_free": vehicle_free,
        "aisle_candidate": extract_row_aisles(vehicle_free, config),
    }
