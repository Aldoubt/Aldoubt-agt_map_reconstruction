"""EXP004-B constant lateral-offset route search inside recovered aisles."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

OCCUPIED_VALUE = np.uint8(0)
UNKNOWN_VALUE = np.uint8(205)
FREE_VALUE = np.uint8(254)


def _footprint(value):
    fp = np.asarray(value, dtype=float)
    if fp.ndim != 2 or fp.shape[1] != 2 or fp.shape[0] < 3:
        raise ValueError('footprint_xy_m must be an Nx2 polygon with at least 3 vertices')
    return fp


def _geometry(rectangle, resolution):
    if resolution <= 0:
        raise ValueError('resolution must be > 0')
    poly = np.asarray(rectangle['polygon_xy'], dtype=float)
    if poly.shape != (4, 2):
        raise ValueError(f"aisle {rectangle.get('aisle_id')} polygon must be 4x2")
    start = 0.5 * (poly[0] + poly[3])
    end = 0.5 * (poly[1] + poly[2])
    delta = end - start
    length_cells = float(np.linalg.norm(delta))
    if length_cells <= 0:
        raise ValueError(f"aisle {rectangle.get('aisle_id')} centerline has zero length")
    unit = delta / length_cells
    normal = np.array([-unit[1], unit[0]], dtype=float)
    widths = [
        abs(float(np.dot(poly[3] - poly[0], normal))),
        abs(float(np.dot(poly[2] - poly[1], normal))),
    ]
    return {
        'poly': poly,
        'start': start,
        'end': end,
        'unit': unit,
        'normal': normal,
        'length_m': length_cells * resolution,
        'width_m': min(widths) * resolution,
        'yaw': math.atan2(unit[1], unit[0]),
    }


def lateral_offset_candidates(rectangle, footprint_xy_m, resolution, offset_step_m=0.05):
    """Return aisle-local +Y offsets whose polygon can fit inside the aisle width."""
    fp = _footprint(footprint_xy_m)
    if offset_step_m <= 0:
        raise ValueError('offset_step_m must be > 0')
    geom = _geometry(rectangle, float(resolution))
    half = geom['width_m'] / 2.0
    low = -half + max(0.0, -float(fp[:, 1].min()))
    high = half - max(0.0, float(fp[:, 1].max()))
    if low > high + 1e-12:
        return []

    values = [0.0] if low <= 0.0 <= high else []
    step = float(offset_step_m)
    k = 1
    while k * step <= high + 1e-12:
        values.append(k * step)
        k += 1
    k = 1
    while -k * step >= low - 1e-12:
        values.append(-k * step)
        k += 1
    values.extend([low, high])
    out = []
    for value in sorted(float(np.clip(v, low, high)) for v in values):
        if not out or abs(value - out[-1]) > 1e-9:
            out.append(value)
    return out


def _sample_centres(rectangle, fp, resolution, spacing, offset):
    if spacing <= 0:
        raise ValueError('sample_spacing_m must be > 0')
    geom = _geometry(rectangle, resolution)
    rear = max(0.0, -float(fp[:, 0].min()))
    front = max(0.0, float(fp[:, 0].max()))
    usable = geom['length_m'] - rear - front
    if usable < -1e-12:
        return [], geom
    count = max(1, int(math.ceil(max(usable, 0.0) / spacing)) + 1)
    distances = np.linspace(rear, geom['length_m'] - front, count)
    offset_cells = geom['normal'] * (offset / resolution)
    centres = [
        geom['start'] + geom['unit'] * (float(d) / resolution) + offset_cells
        for d in distances
    ]
    return centres, geom


def _pose_cells(shape, fp, centre, yaw, resolution):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    poly = np.asarray(centre) + (fp @ rotation.T) / resolution
    h, w = shape
    out = bool(
        np.any(poly[:, 0] < 0) or np.any(poly[:, 0] > w - 1)
        or np.any(poly[:, 1] < 0) or np.any(poly[:, 1] > h - 1)
    )
    pts = np.rint(poly).astype(np.int32)
    min_x, max_x = int(pts[:, 0].min()), int(pts[:, 0].max())
    min_y, max_y = int(pts[:, 1].min()), int(pts[:, 1].max())
    local = np.zeros((max_y - min_y + 1, max_x - min_x + 1), np.uint8)
    cv2.fillPoly(local, [pts - np.array([min_x, min_y], np.int32)], 1)
    yy, xx = np.nonzero(local)
    xx, yy = xx + min_x, yy + min_y
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    return yy[valid], xx[valid], out


def _evaluate(base, rectangle, fp, resolution, spacing, offset, candidate, allow_unknown, distance):
    centres, geom = _sample_centres(rectangle, fp, resolution, spacing, offset)
    if not centres:
        return {
            'offset_m': float(offset), 'passed': False, 'pose_count': 0,
            'blocking_pose_count': 0, 'collision_pose_count': 0,
            'unknown_overlap_pose_count': 0, 'candidate_overlap_pose_count': 0,
            'out_of_bounds_pose_count': 0, 'min_blocked_clearance_m': 0.0,
            'clearance_p10_m': 0.0, 'mean_clearance_m': 0.0,
            'first_failure_reason': 'footprint_longer_than_aisle',
            'first_failure_pose': None,
        }

    counts = {'blocking': 0, 'collision': 0, 'unknown': 0, 'candidate': 0, 'out': 0}
    clearances, first_reason, first_pose = [], None, None
    for centre in centres:
        yy, xx, out = _pose_cells(base.shape, fp, centre, geom['yaw'], resolution)
        if len(xx) == 0:
            out = True
        values = base[yy, xx] if len(xx) else np.asarray([], dtype=np.uint8)
        collision = bool(np.any(values == OCCUPIED_VALUE))
        unknown = bool(np.any(values == UNKNOWN_VALUE))
        cand = bool(candidate is not None and len(xx) and np.any(candidate[yy, xx]))
        blocking = bool(out or collision or (unknown and not allow_unknown))
        clearance = 0.0 if out or not len(xx) else float(np.min(distance[yy, xx]))
        clearances.append(clearance)
        counts['blocking'] += int(blocking)
        counts['collision'] += int(collision)
        counts['unknown'] += int(unknown)
        counts['candidate'] += int(cand)
        counts['out'] += int(out)
        if blocking and first_reason is None:
            first_reason = 'out_of_bounds' if out else ('occupied' if collision else 'unknown')
            first_pose = {'x_cell': float(centre[0]), 'y_cell': float(centre[1]), 'yaw_rad': float(geom['yaw'])}

    return {
        'offset_m': float(offset), 'passed': counts['blocking'] == 0,
        'pose_count': len(centres), 'blocking_pose_count': counts['blocking'],
        'collision_pose_count': counts['collision'],
        'unknown_overlap_pose_count': counts['unknown'],
        'candidate_overlap_pose_count': counts['candidate'],
        'out_of_bounds_pose_count': counts['out'],
        'min_blocked_clearance_m': float(min(clearances)),
        'clearance_p10_m': float(np.percentile(clearances, 10)),
        'mean_clearance_m': float(np.mean(clearances)),
        'first_failure_reason': first_reason, 'first_failure_pose': first_pose,
    }


def _choose_best(results):
    feasible = [item for item in results if item['passed']]
    if not feasible:
        return None
    return min(feasible, key=lambda item: (
        -item['clearance_p10_m'], -item['min_blocked_clearance_m'],
        item['candidate_overlap_pose_count'], abs(item['offset_m']), item['offset_m'],
    ))


def _choose_attempt(results):
    if not results:
        return None
    return min(results, key=lambda item: (
        item['blocking_pose_count'], item['collision_pose_count'],
        item['unknown_overlap_pose_count'], item['out_of_bounds_pose_count'],
        -item['clearance_p10_m'], -item['min_blocked_clearance_m'],
        item['candidate_overlap_pose_count'], abs(item['offset_m']), item['offset_m'],
    ))


def search_constant_offset_routes(base_map, aisle_rectangles, footprint_xy_m,
                                  resolution, sample_spacing_m=0.10,
                                  offset_step_m=0.05, candidate_mask=None,
                                  allow_unknown=False):
    """Search one constant cross-track offset per aisle; this is not a general planner."""
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError('base_map must be a 2D array')
    fp = _footprint(footprint_xy_m)
    candidate = None if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    if candidate is not None and candidate.shape != base.shape:
        raise ValueError('candidate_mask shape must match base_map')
    blocked = base == OCCUPIED_VALUE
    if not allow_unknown:
        blocked |= base == UNKNOWN_VALUE
    distance = ndimage.distance_transform_edt(~blocked) * float(resolution)

    aisles = []
    for rectangle in aisle_rectangles:
        offsets = lateral_offset_candidates(rectangle, fp, float(resolution), float(offset_step_m))
        routes = [
            _evaluate(base, rectangle, fp, float(resolution), float(sample_spacing_m),
                      offset, candidate, bool(allow_unknown), distance)
            for offset in offsets
        ]
        centre = min(routes, key=lambda item: abs(item['offset_m'])) if routes else None
        best, attempt = _choose_best(routes), _choose_attempt(routes)
        geom = _geometry(rectangle, float(resolution))
        label = rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}")
        aisles.append({
            'aisle_id': int(rectangle['aisle_id']), 'label': label,
            'width_m': float(rectangle.get('width_m', geom['width_m'])),
            'length_m': float(rectangle.get('length_m', geom['length_m'])),
            'heading_rad': float(geom['yaw']),
            'offset_min_m': float(offsets[0]) if offsets else None,
            'offset_max_m': float(offsets[-1]) if offsets else None,
            'tested_offset_count': len(routes),
            'feasible_offset_count': sum(int(item['passed']) for item in routes),
            'centerline_passed': bool(centre and centre['passed']),
            'passed': best is not None,
            'route_recovered': bool(best is not None and centre is not None and not centre['passed']),
            'best_offset_m': None if best is None else float(best['offset_m']),
            'best_min_blocked_clearance_m': None if best is None else float(best['min_blocked_clearance_m']),
            'best_clearance_p10_m': None if best is None else float(best['clearance_p10_m']),
            'best_candidate_overlap_pose_count': None if best is None else int(best['candidate_overlap_pose_count']),
            'best_attempt_offset_m': None if attempt is None else float(attempt['offset_m']),
            'best_attempt_blocking_pose_count': None if attempt is None else int(attempt['blocking_pose_count']),
            'best_attempt_collision_pose_count': None if attempt is None else int(attempt['collision_pose_count']),
            'best_attempt_unknown_pose_count': None if attempt is None else int(attempt['unknown_overlap_pose_count']),
            'offset_results': routes,
        })

    passed = [item for item in aisles if item['passed']]
    recovered = [item for item in aisles if item['route_recovered']]
    centre_pass = [item for item in aisles if item['centerline_passed']]
    failed = [item for item in aisles if not item['passed']]
    return {
        'policy': {
            'allow_unknown': bool(allow_unknown), 'sample_spacing_m': float(sample_spacing_m),
            'offset_step_m': float(offset_step_m), 'search_model': 'constant_lateral_offset',
            'route_score': 'max_clearance_p10_then_min_clearance_then_candidate_overlap',
        },
        'summary': {
            'centerline_pass_count': len(centre_pass), 'pass_count': len(passed),
            'recovered_route_count': len(recovered), 'fail_count': len(failed),
            'total_aisles': len(aisles),
            'recovered_aisles': [item['label'] for item in recovered],
            'failed_aisles': [item['label'] for item in failed],
        },
        'aisles': aisles,
    }


def _route_segment(rectangle, fp, resolution, offset):
    geom = _geometry(rectangle, resolution)
    rear = max(0.0, -float(fp[:, 0].min()))
    front = max(0.0, float(fp[:, 0].max()))
    shift = geom['normal'] * (offset / resolution)
    return (
        geom['start'] + geom['unit'] * (rear / resolution) + shift,
        geom['end'] - geom['unit'] * (front / resolution) + shift,
    )


def _image_point(xy, height):
    return int(round(float(xy[0]))), int(round((height - 1) - float(xy[1])))


def _overlay(base, rectangles, fp, resolution, result, focus=None):
    image = cv2.cvtColor(np.flipud(np.asarray(base, np.uint8)), cv2.COLOR_GRAY2BGR)
    by_label = {item['label']: item for item in result['aisles']}
    h = base.shape[0]
    for rectangle in rectangles:
        label = rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}")
        geom, item = _geometry(rectangle, resolution), by_label[label]
        cv2.line(image, _image_point(geom['start'], h), _image_point(geom['end'], h), (0, 180, 255), 1, cv2.LINE_AA)
        offset = item['best_offset_m']
        color = (0, 180, 0)
        if offset is None:
            offset, color = item['best_attempt_offset_m'], (0, 0, 255)
        if offset is not None:
            start, end = _route_segment(rectangle, fp, resolution, offset)
            cv2.line(image, _image_point(start, h), _image_point(end, h), color, 2, cv2.LINE_AA)
    if focus is None:
        return image
    target = next((r for r in rectangles if r.get('label', f"A{int(r['aisle_id']):02d}") == focus), None)
    if target is None:
        raise ValueError(f'focus aisle not found: {focus}')
    poly = np.asarray(target['polygon_xy'], dtype=float)
    x0, x1 = max(0, int(poly[:, 0].min()) - 30), min(base.shape[1] - 1, int(poly[:, 0].max()) + 30)
    y0, y1 = max(0, int(poly[:, 1].min()) - 30), min(base.shape[0] - 1, int(poly[:, 1].max()) + 30)
    iy0, iy1 = (base.shape[0] - 1) - y1, (base.shape[0] - 1) - y0
    return image[iy0:iy1 + 1, x0:x1 + 1]


def write_offset_search_bundle(base_map, aisle_rectangles, footprint_xy_m,
                               output_dir, resolution, sample_spacing_m=0.10,
                               offset_step_m=0.05, candidate_mask=None,
                               allow_unknown=False, footprint_name='robot',
                               focus_aisle='A05'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fp = _footprint(footprint_xy_m)
    result = search_constant_offset_routes(
        base_map, aisle_rectangles, fp, resolution, sample_spacing_m,
        offset_step_m, candidate_mask, allow_unknown,
    )
    result['footprint'] = {'name': str(footprint_name), 'polygon_xy_m': fp.tolist()}
    result['resolution_m'] = float(resolution)
    (output / 'aisle_offset_search.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    fields = [
        'aisle_id', 'label', 'width_m', 'length_m', 'tested_offset_count',
        'feasible_offset_count', 'centerline_passed', 'passed', 'route_recovered',
        'best_offset_m', 'best_min_blocked_clearance_m', 'best_clearance_p10_m',
        'best_candidate_overlap_pose_count', 'best_attempt_offset_m',
        'best_attempt_blocking_pose_count', 'best_attempt_collision_pose_count',
        'best_attempt_unknown_pose_count',
    ]
    with (output / 'aisle_offset_search.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for aisle in result['aisles']:
            writer.writerow({key: aisle.get(key) for key in fields})
    cv2.imwrite(str(output / 'route_overlay.png'), _overlay(base_map, aisle_rectangles, fp, float(resolution), result))
    if focus_aisle:
        cv2.imwrite(str(output / f'{focus_aisle}_route_overlay.png'), _overlay(base_map, aisle_rectangles, fp, float(resolution), result, focus_aisle))
    return result
