import numpy as np


def extract_corridor(traversability, min_width=5):
    """Extract simple corridor mask from traversability grid.

    0 unknown, 1 free, 2 obstacle.
    This is a first geometric baseline before row-aware methods.
    """
    free = traversability == 1
    if free.size == 0:
        return free

    # Remove tiny isolated regions.
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
