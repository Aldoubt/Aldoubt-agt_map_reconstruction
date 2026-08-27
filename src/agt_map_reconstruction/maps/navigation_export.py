"""Navigation-oriented static map export and validation helpers.

The reconstruction pipeline keeps semantic candidates separate from permanent
static obstacles. Arrays use the repository/grid convention (y, x) with a
lower-left map origin; image writers should flip vertically for PGM output.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

OCCUPIED_VALUE = np.uint8(0)
UNKNOWN_VALUE = np.uint8(205)
FREE_VALUE = np.uint8(254)

HARD_LABELS = (2, 4, 6, 7)  # ridge, wall, pillar, confirmed occupied
CANDIDATE_LABELS = (3, 5)  # obstacle candidate, step candidate
PILLAR_LABELS = (6,)


@dataclass(frozen=True)
class NavigationLayers:
    base_map: np.ndarray
    candidate_mask: np.ndarray
    aisle_prior: np.ndarray
    hard_obstacle_mask: np.ndarray


def rasterize_aisles(rectangles, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for aisle in rectangles:
        polygon = np.asarray(aisle['polygon_xy'], dtype=np.float64)
        if polygon.shape != (4, 2):
            raise ValueError(f"aisle {aisle.get('aisle_id')} polygon must be 4x2")
        points = np.rint(polygon).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def build_navigation_layers(semantic_labels, aisle_rectangles):
    semantic = np.asarray(semantic_labels)
    if semantic.ndim != 2:
        raise ValueError('semantic_labels must be a 2D array')

    aisle_prior = rasterize_aisles(aisle_rectangles, semantic.shape)
    hard = np.isin(semantic, HARD_LABELS)
    candidate = np.isin(semantic, CANDIDATE_LABELS)
    free = (semantic == 1) | aisle_prior

    base = np.full(semantic.shape, UNKNOWN_VALUE, dtype=np.uint8)
    base[free] = FREE_VALUE
    # Structural geometry and confirmed occupied evidence have the highest
    # priority and must never be promoted to free by the aisle prior.
    base[hard] = OCCUPIED_VALUE

    return NavigationLayers(
        base_map=base,
        candidate_mask=candidate,
        aisle_prior=aisle_prior,
        hard_obstacle_mask=hard,
    )


def build_map_yaml(image, resolution, origin=(0.0, 0.0, 0.0)):
    return {
        'image': str(image),
        'mode': 'trinary',
        'resolution': float(resolution),
        'origin': [float(v) for v in origin],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }


def _probe_points(rectangle, fraction=0.1):
    polygon = np.asarray(rectangle['polygon_xy'], dtype=float)
    start = 0.5 * (polygon[0] + polygon[3])
    end = 0.5 * (polygon[1] + polygon[2])
    return start + fraction * (end - start), end - fraction * (end - start)


def _point_label(labels, xy):
    x, y = np.rint(xy).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def validate_navigation_map(base_map, aisle_rectangles, resolution,
                            clearance_radii_m=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50)):
    base = np.asarray(base_map)
    if base.ndim != 2:
        raise ValueError('base_map must be a 2D array')

    observed = sorted(int(v) for v in np.unique(base))
    canonical = {int(OCCUPIED_VALUE), int(UNKNOWN_VALUE), int(FREE_VALUE)}
    unexpected = [v for v in observed if v not in canonical]

    free = base == FREE_VALUE
    distance_m = ndimage.distance_transform_edt(free) * float(resolution)

    aisle_results = []
    for rectangle in aisle_rectangles:
        aisle_results.append({
            'aisle_id': int(rectangle['aisle_id']),
            'label': rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}"),
            'width_m': float(rectangle.get('width_m', 0.0)),
            'length_m': float(rectangle.get('length_m', 0.0)),
            'clearance_pass': {},
        })

    clearance_tests = {}
    for radius in clearance_radii_m:
        key = f'{float(radius):.2f}'
        safe = free & (distance_m + 1e-12 >= float(radius))
        pass_count = 0

        for rectangle, result in zip(aisle_rectangles, aisle_results):
            aisle_mask = rasterize_aisles([rectangle], base.shape)
            labels, _ = ndimage.label(safe & aisle_mask)
            start, end = _probe_points(rectangle)
            start_label = _point_label(labels, start)
            end_label = _point_label(labels, end)
            passed = start_label > 0 and start_label == end_label
            result['clearance_pass'][key] = bool(passed)
            pass_count += int(passed)

        clearance_tests[key] = {
            'radius_m': float(radius),
            'diameter_m': 2.0 * float(radius),
            'pass_count': int(pass_count),
            'total_aisles': len(aisle_rectangles),
        }

    return {
        'gray_semantics_valid': not unexpected,
        'observed_gray_values': observed,
        'unexpected_gray_values': unexpected,
        'resolution_m': float(resolution),
        'clearance_tests': clearance_tests,
        'aisles': aisle_results,
    }


def write_pgm(grid, path):
    """Write a uint8 occupancy image as binary PGM using map-server image orientation."""
    from pathlib import Path

    image = np.asarray(grid, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pgm = np.flipud(image)
    with path.open('wb') as stream:
        stream.write(f'P5\n{pgm.shape[1]} {pgm.shape[0]}\n255\n'.encode('ascii'))
        stream.write(pgm.tobytes(order='C'))


def write_navigation_bundle(semantic_labels, aisle_rectangles, output_dir,
                            resolution=0.05, origin=(0.0, 0.0, 0.0),
                            clearance_radii_m=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50)):
    """Build and persist the navigation-map-v2 artifact bundle."""
    import json
    from pathlib import Path

    import yaml

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layers = build_navigation_layers(semantic_labels, aisle_rectangles)
    map_yaml = build_map_yaml('navigation_base_map.pgm', resolution, origin)
    validation = validate_navigation_map(
        layers.base_map,
        aisle_rectangles,
        resolution,
        clearance_radii_m=clearance_radii_m,
    )
    validation.update({
        'map_server_yaml_valid': bool(
            map_yaml['mode'] == 'trinary'
            and 0.0 <= map_yaml['free_thresh'] < map_yaml['occupied_thresh'] <= 1.0
        ),
        'candidate_cell_count': int(layers.candidate_mask.sum()),
        'hard_obstacle_cell_count': int(layers.hard_obstacle_mask.sum()),
        'hard_obstacle_as_free_cell_count': int(
            np.count_nonzero(layers.hard_obstacle_mask & (layers.base_map == FREE_VALUE))
        ),
        'static_obstacle_semantics_valid': bool(
            not np.any(layers.hard_obstacle_mask & (layers.base_map == FREE_VALUE))
        ),
        'pillar_cell_count': int(np.count_nonzero(np.isin(np.asarray(semantic_labels), PILLAR_LABELS))),
        'pillar_as_free_cell_count': int(
            np.count_nonzero(
                np.isin(np.asarray(semantic_labels), PILLAR_LABELS)
                & (layers.base_map == FREE_VALUE)
            )
        ),
        'free_cell_count': int(np.count_nonzero(layers.base_map == FREE_VALUE)),
        'unknown_cell_count': int(np.count_nonzero(layers.base_map == UNKNOWN_VALUE)),
        'occupied_cell_count': int(np.count_nonzero(layers.base_map == OCCUPIED_VALUE)),
    })

    write_pgm(layers.base_map, output / 'navigation_base_map.pgm')
    with (output / 'navigation_base_map.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(map_yaml, stream, sort_keys=False)
    np.save(output / 'candidate_mask.npy', layers.candidate_mask.astype(np.uint8))
    np.save(output / 'static_obstacle_mask.npy', layers.hard_obstacle_mask.astype(np.uint8))
    with (output / 'validation.json').open('w', encoding='utf-8') as stream:
        json.dump(validation, stream, indent=2, sort_keys=True)
        stream.write('\n')

    return {
        'layers': layers,
        'map_yaml': map_yaml,
        'validation': validation,
    }
