"""Bridge conservative ground evidence into the current navigation semantics."""

import numpy as np

from .ground_evidence import EvidenceClass

LABEL_UNKNOWN = np.uint8(0)
LABEL_AISLE = np.uint8(1)
LABEL_RIDGE = np.uint8(2)
LABEL_OBSTACLE_CANDIDATE = np.uint8(3)
LABEL_WALL = np.uint8(4)
LABEL_STEP_CANDIDATE = np.uint8(5)
LABEL_PILLAR = np.uint8(6)
LABEL_OCCUPIED_CONFIRMED = np.uint8(7)


def _as_evidence(evidence):
    value = np.asarray(evidence)
    if value.ndim != 2:
        raise ValueError("evidence must be a 2D array")
    valid = tuple(int(item) for item in EvidenceClass)
    if not np.isin(value, valid).all():
        raise ValueError("evidence contains an unknown label")
    return value.astype(np.uint8, copy=False)


def semantic_labels_from_evidence(evidence):
    """Convert four-state ground evidence to EXP003 navigation semantics.

    Confirmed free becomes semantic free. Confirmed occupied initially receives
    its own hard evidence label instead of being misnamed as ridge/wall/pillar.
    Interpolated ground stays unknown in the static semantic grid and is only
    available to corridor-geometry recovery as an optional continuity hint.
    """
    evidence = _as_evidence(evidence)
    labels = np.full(evidence.shape, LABEL_UNKNOWN, dtype=np.uint8)
    labels[evidence == EvidenceClass.FREE_CONFIRMED] = LABEL_AISLE
    labels[evidence == EvidenceClass.OCCUPIED_CONFIRMED] = LABEL_OCCUPIED_CONFIRMED
    return labels


def refine_occupied_evidence_with_aisle_prior(semantic_labels, aisle_prior):
    """Demote measured occupied/aisle conflicts to advisory candidates.

    The recovered aisle prior is allowed to reinterpret only measured
    ``LABEL_OCCUPIED_CONFIRMED`` cells. Unknown and interpolated cells are not
    promoted, and confirmed occupied evidence outside recovered aisles remains
    hard. This separates low-level elevated returns from persistent structural
    semantics without erasing the original evidence or candidate mask.
    """
    semantic = np.asarray(semantic_labels)
    aisle = np.asarray(aisle_prior, dtype=bool)
    if semantic.ndim != 2 or aisle.ndim != 2:
        raise ValueError("semantic_labels and aisle_prior must be 2D arrays")
    if semantic.shape != aisle.shape:
        raise ValueError("semantic_labels and aisle_prior shape must match")

    refined = semantic.astype(np.uint8, copy=True)
    conflict = aisle & (semantic == LABEL_OCCUPIED_CONFIRMED)
    refined[conflict] = LABEL_OBSTACLE_CANDIDATE
    return refined


def corridor_seed_from_evidence(evidence, include_interpolated=True):
    """Return cells that may support longitudinal aisle-geometry recovery."""
    evidence = _as_evidence(evidence)
    seed = evidence == EvidenceClass.FREE_CONFIRMED
    if include_interpolated:
        seed |= evidence == EvidenceClass.GROUND_INTERPOLATED
    return seed


def build_basic_semantic_labels(traversability, corridor):
    """Fallback baseline for legacy EXP002 products; prefer ground evidence.

    Recovered corridor cells become free, while relative-height obstacles stay
    advisory candidates. Traversable cells outside a recovered corridor remain
    unknown because EXP002's binary mask was not accepted as global free truth.
    """
    traversability = np.asarray(traversability)
    corridor = np.asarray(corridor, dtype=bool)
    if traversability.ndim != 2 or corridor.ndim != 2:
        raise ValueError("traversability and corridor must be 2D arrays")
    if traversability.shape != corridor.shape:
        raise ValueError("traversability and corridor shape must match")
    labels = np.full(traversability.shape, LABEL_UNKNOWN, dtype=np.uint8)
    labels[corridor] = LABEL_AISLE
    labels[traversability == 2] = LABEL_OBSTACLE_CANDIDATE
    return labels
