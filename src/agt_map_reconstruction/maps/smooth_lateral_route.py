"""EXP004-B2 smooth lateral route search inside recovered agricultural aisles.

The planner is intentionally an offline geometry validator rather than a general
navigation planner. It searches a lattice of lateral offsets along the recovered
aisle axis, allows the offset to change gradually, and validates every interpolated
pose with the measured polygon footprint.
"""
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


def _as_footprint(value):
    fp = np.asarray(value, dtype=float)
    if fp.ndim != 2 or fp.shape[1] != 2 or fp.shape[0] < 3:
        raise ValueError('footprint_xy_m must be an Nx2 polygon with at least 3 vertices')
    return fp


def _geometry(rectangle, resolution):
    if resolution <= 0.0:
        raise ValueError('resolution must be > 0')
    poly = np.asarray(rectangle['polygon_xy'], dtype=float)
    if poly.shape != (4, 2):
        raise ValueError(f"aisle {rectangle.get('aisle_id')} polygon must be 4x2")
    start = 0.5 * (poly[0] + poly[3])
    end = 0.5 * (poly[1] + poly[2])
    delta = end - start
    length_cells = float(np.linalg.norm(delta))
    if length_cells <= 0.0:
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


def _offset_candidates(rectangle, footprint, resolution, offset_step_m):
    if offset_step_m <= 0.0:
        raise ValueError('offset_step_m must be > 0')
    geom = _geometry(rectangle, resolution)
    half = 0.5 * geom['width_m']
    low = -half + max(0.0, -float(footprint[:, 1].min()))
    high = half - max(0.0, float(footprint[:, 1].max()))
    if low > high + 1e-12:
        return []
    values = [low, high]
    if low <= 0.0 <= high:
        values.append(0.0)
    k = 1
    while k * offset_step_m <= high + 1e-12:
        values.append(k * offset_step_m)
        k += 1
    k = 1
    while -k * offset_step_m >= low - 1e-12:
        values.append(-k * offset_step_m)
        k += 1
    out = []
    for value in sorted(float(np.clip(v, low, high)) for v in values):
        if not out or abs(value - out[-1]) > 1e-9:
            out.append(value)
    return out


def _pose_cells(shape, footprint, centre, yaw, resolution):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    polygon = np.asarray(centre, dtype=float) + (footprint @ rotation.T) / resolution
    h, w = shape
    out = bool(
        np.any(polygon[:, 0] < 0.0) or np.any(polygon[:, 0] > w - 1)
        or np.any(polygon[:, 1] < 0.0) or np.any(polygon[:, 1] > h - 1)
    )
    pts = np.rint(polygon).astype(np.int32)
    min_x, max_x = int(pts[:, 0].min()), int(pts[:, 0].max())
    min_y, max_y = int(pts[:, 1].min()), int(pts[:, 1].max())
    local = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    cv2.fillPoly(local, [pts - np.array([min_x, min_y], dtype=np.int32)], 1)
    yy, xx = np.nonzero(local)
    xx, yy = xx + min_x, yy + min_y
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    return yy[valid], xx[valid], out


def _stations(geom, footprint, control_spacing_m, endpoint_trim_m=0.0):
    if control_spacing_m <= 0.0:
        raise ValueError('control_spacing_m must be > 0')
    if endpoint_trim_m < 0.0:
        raise ValueError('endpoint_trim_m must be >= 0')
    rear = max(0.0, -float(footprint[:, 0].min())) + float(endpoint_trim_m)
    front = max(0.0, float(footprint[:, 0].max())) + float(endpoint_trim_m)
    usable = geom['length_m'] - rear - front
    if usable < -1e-12:
        return []
    count = max(2, int(math.ceil(max(usable, 0.0) / control_spacing_m)) + 1)
    return np.linspace(rear, geom['length_m'] - front, count).tolist()


def _centre_at(geom, distance_m, offset_m, resolution):
    return (
        geom['start']
        + geom['unit'] * (float(distance_m) / resolution)
        + geom['normal'] * (float(offset_m) / resolution)
    )


def _swept_edge(base_shape, blocked_mask, geom, footprint, resolution,
                d0, d1, o0, o1, candidate, blocked_distance):
    """Continuously validate a convex footprint swept along one linear edge."""
    ds = float(d1 - d0)
    heading_delta = math.atan2(float(o1 - o0), ds)
    yaw = geom['yaw'] + heading_delta
    c, ss = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -ss], [ss, c]], dtype=float)
    relative = (footprint @ rotation.T) / resolution
    c0 = _centre_at(geom, d0, o0, resolution)
    c1 = _centre_at(geom, d1, o1, resolution)
    points = np.vstack([c0 + relative, c1 + relative])
    h, w = base_shape
    out = bool(
        np.any(points[:, 0] < 0.0) or np.any(points[:, 0] > w - 1)
        or np.any(points[:, 1] < 0.0) or np.any(points[:, 1] > h - 1)
    )
    pts = np.rint(points).astype(np.int32)
    hull = cv2.convexHull(pts)
    min_x, max_x = int(hull[:, 0, 0].min()), int(hull[:, 0, 0].max())
    min_y, max_y = int(hull[:, 0, 1].min()), int(hull[:, 0, 1].max())
    local = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    shifted = hull.copy()
    shifted[:, 0, 0] -= min_x
    shifted[:, 0, 1] -= min_y
    cv2.fillConvexPoly(local, shifted, 1)
    yy, xx = np.nonzero(local)
    xx, yy = xx + min_x, yy + min_y
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    xx, yy = xx[valid], yy[valid]
    blocked = bool(out or len(xx) == 0 or np.any(blocked_mask[yy, xx]))
    cand_cells = int(np.count_nonzero(candidate[yy, xx])) if candidate is not None and len(xx) else 0
    clearances = blocked_distance[yy, xx] if len(xx) else np.asarray([0.0])
    return {
        'passed': not blocked,
        'candidate_cell_count': cand_cells,
        'min_clearance_m': float(np.min(clearances)),
        'clearance_p10_m': float(np.percentile(clearances, 10)),
        'heading_deviation_rad': float(abs(heading_delta)),
    }


def _sample_edge(base, geom, footprint, resolution, sample_spacing_m,
                 d0, d1, o0, o1, candidate, allow_unknown, blocked_distance):
    ds = float(d1 - d0)
    if ds <= 0.0:
        raise ValueError('control stations must be strictly increasing')
    count = max(2, int(math.ceil(ds / sample_spacing_m)) + 1)
    distances = np.linspace(d0, d1, count)
    offsets = np.linspace(o0, o1, count)
    heading_delta = math.atan2(float(o1 - o0), ds)
    yaw = geom['yaw'] + heading_delta

    points = []
    clearances = []
    collision_count = 0
    unknown_count = 0
    candidate_count = 0
    out_count = 0
    blocking_count = 0

    for distance_m, offset_m in zip(distances, offsets):
        centre = (
            geom['start']
            + geom['unit'] * (float(distance_m) / resolution)
            + geom['normal'] * (float(offset_m) / resolution)
        )
        yy, xx, out = _pose_cells(base.shape, footprint, centre, yaw, resolution)
        if len(xx) == 0:
            out = True
        values = base[yy, xx] if len(xx) else np.asarray([], dtype=np.uint8)
        collision = bool(np.any(values == OCCUPIED_VALUE))
        unknown = bool(np.any(values == UNKNOWN_VALUE))
        cand = bool(candidate is not None and len(xx) and np.any(candidate[yy, xx]))
        blocking = bool(out or collision or (unknown and not allow_unknown))
        clearance = 0.0 if out or not len(xx) else float(np.min(blocked_distance[yy, xx]))
        clearances.append(clearance)
        collision_count += int(collision)
        unknown_count += int(unknown)
        candidate_count += int(cand)
        out_count += int(out)
        blocking_count += int(blocking)
        points.append({
            'distance_m': float(distance_m),
            'offset_m': float(offset_m),
            'x_cell': float(centre[0]),
            'y_cell': float(centre[1]),
            'yaw_rad': float(yaw),
            'clearance_m': float(clearance),
        })

    return {
        'passed': blocking_count == 0,
        'blocking_pose_count': int(blocking_count),
        'collision_pose_count': int(collision_count),
        'unknown_overlap_pose_count': int(unknown_count),
        'candidate_overlap_pose_count': int(candidate_count),
        'out_of_bounds_pose_count': int(out_count),
        'min_clearance_m': float(min(clearances)),
        'clearance_p10_m': float(np.percentile(clearances, 10)),
        'heading_deviation_rad': float(abs(heading_delta)),
        'points': points,
    }


def _baseline_map(baseline_b1):
    if baseline_b1 is None:
        return {}
    return {item['label']: item for item in baseline_b1.get('aisles', [])}


def _search_one(base, rectangle, footprint, resolution, sample_spacing_m,
                control_spacing_m, offset_step_m, max_offset_change_m, endpoint_trim_m,
                candidate, allow_unknown, blocked_mask, blocked_distance, baseline_item):
    if sample_spacing_m <= 0.0:
        raise ValueError('sample_spacing_m must be > 0')
    if max_offset_change_m < 0.0:
        raise ValueError('max_offset_change_m must be >= 0')

    geom = _geometry(rectangle, resolution)
    offsets = _offset_candidates(rectangle, footprint, resolution, offset_step_m)
    stations = _stations(geom, footprint, control_spacing_m, endpoint_trim_m)
    label = rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}")
    baseline_available = baseline_item is not None
    b1_passed = None if not baseline_available else bool(baseline_item.get('passed'))

    empty = {
        'aisle_id': int(rectangle['aisle_id']), 'label': label,
        'width_m': float(rectangle.get('width_m', geom['width_m'])),
        'length_m': float(rectangle.get('length_m', geom['length_m'])),
        'passed': False, 'b1_passed': b1_passed,
        'baseline_b1_available': baseline_available,
        'route_recovered_from_b1': False,
        'control_point_count': len(stations), 'tested_offset_count': len(offsets),
        'control_offsets_m': [], 'route_points': [],
        'blocking_pose_count': 0, 'collision_pose_count': 0,
        'unknown_overlap_pose_count': 0, 'candidate_overlap_pose_count': 0,
        'out_of_bounds_pose_count': 0,
        'min_blocked_clearance_m': None, 'clearance_p10_m': None,
        'max_offset_step_m': None, 'max_heading_deviation_rad': None,
        'failure_reason': None, 'failure_region': None,
    }
    if not offsets:
        empty['failure_reason'] = 'footprint_wider_than_aisle'
        empty['failure_region'] = 'aisle_geometry'
        return empty
    if len(stations) < 2:
        empty['failure_reason'] = 'footprint_longer_than_aisle'
        empty['failure_region'] = 'aisle_geometry'
        return empty

    edge_cache = {}

    def edge(stage, i, j):
        key = (stage, i, j)
        if key not in edge_cache:
            edge_cache[key] = _swept_edge(
                base.shape, blocked_mask, geom, footprint, resolution,
                stations[stage], stations[stage + 1], offsets[i], offsets[j],
                candidate, blocked_distance,
            )
        return edge_cache[key]

    states = {}
    for i in range(len(offsets)):
        for j in range(len(offsets)):
            delta = offsets[j] - offsets[i]
            if abs(delta) > max_offset_change_m + 1e-12:
                continue
            e = edge(0, i, j)
            if not e['passed']:
                continue
            clearance_cost = 0.25 / max(e['clearance_p10_m'], resolution)
            cost = (
                clearance_cost
                + 0.20 * (abs(offsets[i]) + abs(offsets[j]))
                + 1.50 * abs(delta)
                + 0.005 * e['candidate_cell_count']
            )
            states[(i, j)] = (float(cost), [i, j])

    if not states:
        empty['failure_reason'] = 'no_feasible_first_transition'
        empty['failure_region'] = 'entry'
        return empty

    for stage in range(1, len(stations) - 1):
        next_states = {}
        for (i, j), (cost, path) in states.items():
            prev_delta = offsets[j] - offsets[i]
            for k in range(len(offsets)):
                delta = offsets[k] - offsets[j]
                if abs(delta) > max_offset_change_m + 1e-12:
                    continue
                e = edge(stage, j, k)
                if not e['passed']:
                    continue
                clearance_cost = 0.25 / max(e['clearance_p10_m'], resolution)
                curvature_proxy = abs(delta - prev_delta)
                new_cost = (
                    cost + clearance_cost
                    + 0.20 * abs(offsets[k])
                    + 1.50 * abs(delta)
                    + 4.00 * curvature_proxy
                    + 0.005 * e['candidate_cell_count']
                )
                key = (j, k)
                old = next_states.get(key)
                if old is None or new_cost < old[0]:
                    next_states[key] = (float(new_cost), path + [k])
        states = next_states
        if not states:
            empty['failure_reason'] = f'no_feasible_transition_at_station_{stage + 1}'
            empty['failure_region'] = 'exit' if stage + 1 == len(stations) - 1 else 'interior'
            return empty

    _, path_indices = min(states.values(), key=lambda item: item[0])
    control_offsets = [float(offsets[i]) for i in path_indices]

    route_points = []
    edge_metrics = []
    for stage in range(len(stations) - 1):
        e = _sample_edge(
            base, geom, footprint, resolution, sample_spacing_m,
            stations[stage], stations[stage + 1],
            offsets[path_indices[stage]], offsets[path_indices[stage + 1]],
            candidate, allow_unknown, blocked_distance,
        )
        edge_metrics.append(e)
        pts = e['points'] if stage == 0 else e['points'][1:]
        route_points.extend(pts)

    clearances = [p['clearance_m'] for p in route_points]
    deltas = np.abs(np.diff(np.asarray(control_offsets, dtype=float)))
    max_step = float(deltas.max()) if len(deltas) else 0.0
    max_heading = max((e['heading_deviation_rad'] for e in edge_metrics), default=0.0)
    collision = sum(e['collision_pose_count'] for e in edge_metrics)
    unknown = sum(e['unknown_overlap_pose_count'] for e in edge_metrics)
    candidate_count = sum(e['candidate_overlap_pose_count'] for e in edge_metrics)
    out_count = sum(e['out_of_bounds_pose_count'] for e in edge_metrics)
    blocking = sum(e['blocking_pose_count'] for e in edge_metrics)

    return {
        **empty,
        'passed': True,
        'route_recovered_from_b1': bool(baseline_available and not b1_passed),
        'control_offsets_m': control_offsets,
        'route_points': route_points,
        'blocking_pose_count': int(blocking),
        'collision_pose_count': int(collision),
        'unknown_overlap_pose_count': int(unknown),
        'candidate_overlap_pose_count': int(candidate_count),
        'out_of_bounds_pose_count': int(out_count),
        'min_blocked_clearance_m': float(min(clearances)),
        'clearance_p10_m': float(np.percentile(clearances, 10)),
        'max_offset_step_m': max_step,
        'max_heading_deviation_rad': float(max_heading),
        'failure_reason': None,
    }


def search_smooth_lateral_routes(base_map, aisle_rectangles, footprint_xy_m,
                                 resolution, sample_spacing_m=0.10,
                                 control_spacing_m=0.50, offset_step_m=0.05,
                                 max_offset_change_m=0.10, endpoint_trim_m=0.0,
                                 candidate_mask=None, allow_unknown=False, baseline_b1=None):
    """Search a smooth, spatially varying lateral route for each recovered aisle."""
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError('base_map must be a 2D array')
    footprint = _as_footprint(footprint_xy_m)
    candidate = None if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    if candidate is not None and candidate.shape != base.shape:
        raise ValueError('candidate_mask shape must match base_map')
    blocked = base == OCCUPIED_VALUE
    if not allow_unknown:
        blocked |= base == UNKNOWN_VALUE
    blocked_distance = ndimage.distance_transform_edt(~blocked) * float(resolution)
    baseline = _baseline_map(baseline_b1)

    aisles = [
        _search_one(
            base, rectangle, footprint, float(resolution), float(sample_spacing_m),
            float(control_spacing_m), float(offset_step_m), float(max_offset_change_m),
            float(endpoint_trim_m), candidate, bool(allow_unknown), blocked,
            blocked_distance,
            baseline.get(rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}")),
        )
        for rectangle in aisle_rectangles
    ]
    passed = [a for a in aisles if a['passed']]
    failed = [a for a in aisles if not a['passed']]
    baseline_pass = [a for a in aisles if a['b1_passed'] is True]
    recovered = [a for a in aisles if a['route_recovered_from_b1']]
    return {
        'policy': {
            'search_model': 'smooth_lateral_offset_lattice',
            'allow_unknown': bool(allow_unknown),
            'sample_spacing_m': float(sample_spacing_m),
            'control_spacing_m': float(control_spacing_m),
            'offset_step_m': float(offset_step_m),
            'max_offset_change_m': float(max_offset_change_m),
            'endpoint_trim_m': float(endpoint_trim_m),
            'kinematic_model': 'none_ackermann_deferred',
        },
        'summary': {
            'baseline_b1_available': bool(baseline),
            'baseline_b1_pass_count': len(baseline_pass),
            'pass_count': len(passed),
            'recovered_from_b1_count': len(recovered),
            'fail_count': len(failed),
            'total_aisles': len(aisles),
            'recovered_from_b1_aisles': [a['label'] for a in recovered],
            'failed_aisles': [a['label'] for a in failed],
        },
        'aisles': aisles,
    }


def _image_point(point, height):
    return int(round(float(point[0]))), int(round((height - 1) - float(point[1])))


def _draw_legend(image):
    x, y = 12, 18
    entries = [
        ((0, 180, 255), 'recovered centerline'),
        ((0, 180, 0), 'smooth route PASS'),
        ((0, 0, 255), 'B1 best failed constant route'),
    ]
    cv2.rectangle(image, (6, 4), (245, 66), (245, 245, 245), -1)
    cv2.rectangle(image, (6, 4), (245, 66), (40, 40, 40), 1)
    for idx, (color, text) in enumerate(entries):
        yy = y + idx * 18
        cv2.line(image, (x, yy), (x + 24, yy), color, 2, cv2.LINE_AA)
        cv2.putText(image, text, (x + 32, yy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (20, 20, 20), 1, cv2.LINE_AA)


def _constant_segment(rectangle, footprint, resolution, offset):
    geom = _geometry(rectangle, resolution)
    rear = max(0.0, -float(footprint[:, 0].min()))
    front = max(0.0, float(footprint[:, 0].max()))
    shift = geom['normal'] * (offset / resolution)
    return (
        geom['start'] + geom['unit'] * (rear / resolution) + shift,
        geom['end'] - geom['unit'] * (front / resolution) + shift,
    )


def _overlay(base, rectangles, footprint, resolution, result, baseline_b1=None, focus=None):
    image = cv2.cvtColor(np.flipud(np.asarray(base, dtype=np.uint8)), cv2.COLOR_GRAY2BGR)
    h = base.shape[0]
    by_label = {a['label']: a for a in result['aisles']}
    baseline = _baseline_map(baseline_b1)
    for rectangle in rectangles:
        label = rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}")
        geom = _geometry(rectangle, resolution)
        item = by_label[label]
        cv2.line(image, _image_point(geom['start'], h), _image_point(geom['end'], h),
                 (0, 180, 255), 1, cv2.LINE_AA)
        if item['passed'] and item['route_points']:
            pts = np.asarray([
                _image_point((p['x_cell'], p['y_cell']), h) for p in item['route_points']
            ], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(image, [pts], False, (0, 180, 0), 2, cv2.LINE_AA)
        elif label in baseline:
            b = baseline[label]
            offset = b.get('best_attempt_offset_m')
            if offset is not None:
                start, end = _constant_segment(rectangle, footprint, resolution, float(offset))
                cv2.line(image, _image_point(start, h), _image_point(end, h),
                         (0, 0, 255), 2, cv2.LINE_AA)
    _draw_legend(image)

    if focus is None:
        return image
    target = next((r for r in rectangles if r.get('label', f"A{int(r['aisle_id']):02d}") == focus), None)
    if target is None:
        raise ValueError(f'focus aisle not found: {focus}')
    poly = np.asarray(target['polygon_xy'], dtype=float)
    x0 = max(0, int(poly[:, 0].min()) - 50)
    x1 = min(base.shape[1] - 1, int(poly[:, 0].max()) + 50)
    y0 = max(0, int(poly[:, 1].min()) - 50)
    y1 = min(base.shape[0] - 1, int(poly[:, 1].max()) + 50)
    iy0, iy1 = (base.shape[0] - 1) - y1, (base.shape[0] - 1) - y0
    crop = image[iy0:iy1 + 1, x0:x1 + 1].copy()
    _draw_legend(crop)
    item = by_label[focus]
    status = 'PASS' if item['passed'] else 'FAIL'
    extra = ''
    if item['passed']:
        extra = f"  max_step={item['max_offset_step_m']:.2f}m"
    cv2.rectangle(crop, (4, max(4, crop.shape[0] - 28)),
                  (min(crop.shape[1] - 4, 260), crop.shape[0] - 4),
                  (245, 245, 245), -1)
    cv2.putText(crop, f'{focus} {status}{extra}', (10, crop.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return crop


def write_smooth_route_bundle(base_map, aisle_rectangles, footprint_xy_m,
                              output_dir, resolution, sample_spacing_m=0.10,
                              control_spacing_m=0.50, offset_step_m=0.05,
                              max_offset_change_m=0.10, endpoint_trim_m=0.0,
                              candidate_mask=None, allow_unknown=False,
                              baseline_b1=None, footprint_name='robot', focus_aisles=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    footprint = _as_footprint(footprint_xy_m)
    result = search_smooth_lateral_routes(
        base_map, aisle_rectangles, footprint, resolution,
        sample_spacing_m, control_spacing_m, offset_step_m,
        max_offset_change_m, endpoint_trim_m, candidate_mask, allow_unknown, baseline_b1,
    )
    result['footprint'] = {'name': str(footprint_name), 'polygon_xy_m': footprint.tolist()}
    result['resolution_m'] = float(resolution)

    (output / 'smooth_route_search.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    fields = [
        'aisle_id', 'label', 'width_m', 'length_m', 'baseline_b1_available',
        'b1_passed', 'passed', 'route_recovered_from_b1', 'control_point_count',
        'tested_offset_count', 'blocking_pose_count', 'collision_pose_count',
        'unknown_overlap_pose_count', 'candidate_overlap_pose_count',
        'out_of_bounds_pose_count', 'min_blocked_clearance_m', 'clearance_p10_m',
        'max_offset_step_m', 'max_heading_deviation_rad', 'failure_reason', 'failure_region',
    ]
    with (output / 'smooth_route_search.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for aisle in result['aisles']:
            writer.writerow({key: aisle.get(key) for key in fields})

    cv2.imwrite(str(output / 'smooth_route_overlay.png'), _overlay(
        base_map, aisle_rectangles, footprint, float(resolution), result, baseline_b1
    ))
    for label in focus_aisles or []:
        cv2.imwrite(str(output / f'{label}_smooth_route_overlay.png'), _overlay(
            base_map, aisle_rectangles, footprint, float(resolution), result, baseline_b1, label
        ))
    return result
