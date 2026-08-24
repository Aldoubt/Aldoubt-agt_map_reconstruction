import numpy as np


def extract_corridor(traversability, row_direction=None, min_width=5):
    """Extract agricultural corridor candidates.

    Baseline mode:
        connected traversable regions.

    Row-aware mode:
        keeps the interface for future agricultural constraints.
        The current implementation preserves the geometric baseline while
        allowing direction-aware filtering to be added incrementally.

    Args:
        traversability: 0 unknown, 1 free, 2 obstacle.
        row_direction: optional dominant crop-row direction vector.
        min_width: minimum connected component size.
    """
    free = traversability == 1
    if free.size == 0:
        return free

    from scipy import ndimage

    labels, count = ndimage.label(free)
    result = np.zeros_like(free, dtype=bool)

    for i in range(1, count + 1):
        region = labels == i
        if region.sum() >= min_width:
            result |= region

    return result


def skeletonize_corridor(mask):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask)
    except ImportError:
        return mask
