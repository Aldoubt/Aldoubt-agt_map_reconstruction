import numpy as np


def _connected_regions(mask, min_width):
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    result = np.zeros_like(mask, dtype=bool)

    for i in range(1, count + 1):
        region = labels == i
        if region.sum() >= min_width:
            result |= region

    return result


def _direction_consistency(mask, direction):
    """Estimate whether the free region follows the dominant row direction.

    This is a lightweight geometric constraint. It does not perform semantic
    recognition; it only filters structures inconsistent with the recovered
    agricultural orientation.
    """
    if direction is None:
        return np.ones_like(mask, dtype=float)

    dx, dy = direction
    angle = np.arctan2(dy, dx)

    yy, xx = np.indices(mask.shape)
    coords = np.column_stack((xx[mask], yy[mask]))

    if len(coords) < 2:
        return np.zeros_like(mask, dtype=float)

    centered = coords - coords.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    local = vectors[:, np.argmax(values)]

    consistency = abs(np.dot(local, direction))
    return np.full_like(mask, consistency, dtype=float)


def extract_corridor(traversability, row_direction=None, min_width=5,
                     direction_threshold=0.7):
    """Extract agricultural corridor candidates.

    Modes:
        row_direction=None:
            baseline connected free-space extraction.

        row_direction provided:
            applies a lightweight agricultural row consistency score.

    The method remains geometry-only and intentionally avoids learned
    semantic segmentation.
    """
    free = traversability == 1
    if free.size == 0:
        return free

    candidate = _connected_regions(free, min_width)

    if row_direction is None:
        return candidate

    score = _direction_consistency(candidate, row_direction)
    return candidate & (score >= direction_threshold)


def skeletonize_corridor(mask):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask)
    except ImportError:
        return mask
