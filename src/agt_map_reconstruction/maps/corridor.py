import numpy as np


def _component_geometry(region):
    y, x = np.where(region)
    coords = np.column_stack((x, y)).astype(float)
    centered = coords - coords.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, np.argmax(values)]
    normal = np.array([-direction[1], direction[0]])
    along = centered @ direction
    across = centered @ normal
    length = float(np.ptp(along) + 1.0)
    width = float(np.ptp(across) + 1.0)
    return direction, length, width


def _row_frame_angle(row_direction):
    direction = np.asarray(row_direction, dtype=float)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("row_direction must be finite and non-zero")
    direction /= norm
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _rotate_to_row_frame(array, row_direction):
    from scipy import ndimage

    # Image Y grows downward. Rotating by the PCA angle (not its negative)
    # aligns the grid-coordinate direction with the image X axis.
    return ndimage.rotate(
        array,
        _row_frame_angle(row_direction),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def _rotate_from_row_frame(array, row_direction):
    from scipy import ndimage

    return ndimage.rotate(
        array,
        -_row_frame_angle(row_direction),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def restrict_to_row_extent(mask, row_structure, row_direction):
    """Remove headlands outside the longitudinal support of crop rows."""
    mask = np.asarray(mask, dtype=bool)
    row_structure = np.asarray(row_structure, dtype=bool)
    if mask.shape != row_structure.shape:
        raise ValueError("mask and row_structure shapes must match")
    aligned_mask = _rotate_to_row_frame(mask, row_direction).astype(bool)
    aligned_rows = _rotate_to_row_frame(row_structure, row_direction).astype(bool)
    supported_columns = np.flatnonzero(aligned_rows.any(axis=0))
    if supported_columns.size == 0:
        return np.zeros_like(mask)
    start = int(supported_columns[0])
    stop = int(supported_columns[-1]) + 1
    aligned_mask[:, :start] = False
    aligned_mask[:, stop:] = False
    return _rotate_from_row_frame(aligned_mask, row_direction).astype(bool) & mask


def filter_row_aligned_components(mask, row_direction, min_cells=20,
                                  min_length_cells=40,
                                  min_aspect_ratio=3.0,
                                  direction_threshold=0.85,
                                  max_longitudinal_gap_cells=1):
    """Keep elongated components whose PCA axis follows the crop rows."""
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    _row_frame_angle(row_direction)
    aligned = _rotate_to_row_frame(mask, row_direction).astype(bool)
    if max_longitudinal_gap_cells > 0:
        aligned_for_components = ndimage.binary_closing(
            aligned,
            structure=np.ones(
                (1, 2 * max_longitudinal_gap_cells + 1),
                dtype=bool,
            ),
        )
    else:
        aligned_for_components = aligned

    labels, count = ndimage.label(
        aligned_for_components,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    accepted_aligned = np.zeros_like(aligned)
    for label, component_slice in enumerate(ndimage.find_objects(labels), 1):
        if component_slice is None:
            continue
        component_labels = labels[component_slice]
        region = component_labels == label
        if int(region.sum()) < min_cells:
            continue
        local, length, width = _component_geometry(region)
        aspect_ratio = length / max(width, 1.0)
        if (
            length >= min_length_cells
            and aspect_ratio >= min_aspect_ratio
            and abs(float(local[0])) >= direction_threshold
        ):
            accepted_aligned[component_slice] |= region

    accepted_aligned &= aligned
    result = _rotate_from_row_frame(accepted_aligned, row_direction).astype(bool)
    return result & mask


def _true_runs(values):
    padded = np.pad(np.asarray(values, dtype=np.int8), 1)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts, stops))


def _fill_short_gaps(values, max_gap_cells):
    result = np.asarray(values, dtype=bool).copy()
    if max_gap_cells <= 0:
        return result
    for start, stop in _true_runs(~result):
        if start > 0 and stop < len(result) and stop - start <= max_gap_cells:
            result[start:stop] = True
    return result


def extract_parallel_corridors(relative_height, traversability,
                               row_direction, resolution=0.05,
                               row_height_threshold=0.20,
                               min_width_m=0.60, max_width_m=2.00,
                               min_length_m=3.00,
                               min_row_coverage=0.70,
                               min_row_profile=0.25,
                               max_boundary_gap_m=0.30):
    """Recover row-flanked valleys in a coordinate frame aligned to rows."""
    from scipy import ndimage

    relative_height = np.asarray(relative_height, dtype=float)
    traversability = np.asarray(traversability)
    if relative_height.shape != traversability.shape:
        raise ValueError("relative_height and traversability shapes must match")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    angle_deg = _row_frame_angle(row_direction)

    row_structure = (
        np.isfinite(relative_height)
        & (relative_height >= row_height_threshold)
    ) | (traversability == 2)
    free = traversability == 1

    rotated_rows = _rotate_to_row_frame(row_structure, row_direction).astype(bool)
    rotated_free = _rotate_to_row_frame(free, row_direction).astype(bool)

    row_profile = rotated_rows.mean(axis=1)
    smoothed_profile = ndimage.uniform_filter1d(row_profile, size=3)
    row_bands = _true_runs(smoothed_profile >= min_row_profile)
    min_width_cells = max(1, int(np.ceil(min_width_m / resolution)))
    max_width_cells = max(min_width_cells, int(np.floor(max_width_m / resolution)))
    min_length_cells = max(1, int(np.ceil(min_length_m / resolution)))
    max_boundary_gap_cells = max(
        0,
        int(np.floor(max_boundary_gap_m / resolution)),
    )

    candidate = np.zeros_like(rotated_rows)
    widths_m = []
    accepted = 0
    for pair_index, ((_, first_stop), (second_start, _)) in enumerate(
        zip(row_bands, row_bands[1:])
    ):
        gap_start = first_stop
        gap_stop = second_start
        width_cells = gap_stop - gap_start
        if not min_width_cells <= width_cells <= max_width_cells:
            continue

        first_band = rotated_rows[row_bands[pair_index][0]:first_stop]
        second_band = rotated_rows[
            second_start:row_bands[pair_index + 1][1]
        ]
        longitudinal_support = first_band.any(axis=0) & second_band.any(axis=0)
        valid_support = np.zeros_like(longitudinal_support)
        continuous_support = _fill_short_gaps(
            longitudinal_support,
            max_boundary_gap_cells,
        )
        for start, stop in _true_runs(continuous_support):
            run_length = stop - start
            continuity = float(longitudinal_support[start:stop].mean())
            if run_length >= min_length_cells and continuity >= min_row_coverage:
                valid_support[start:stop] = True

        band_candidate = rotated_free[gap_start:gap_stop] & valid_support[None, :]
        if band_candidate.any():
            candidate[gap_start:gap_stop] |= band_candidate
            widths_m.append(float(width_cells * resolution))
            accepted += 1

    corridor = _rotate_from_row_frame(candidate, row_direction).astype(bool)
    corridor &= free
    details = {
        "accepted_corridors": accepted,
        "widths_m": widths_m,
        "row_profile_peak_count": len(row_bands),
        "min_row_profile": float(min_row_profile),
        "max_boundary_gap_m": float(max_boundary_gap_m),
        "row_frame_angle_deg": angle_deg,
    }
    return corridor, details


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
