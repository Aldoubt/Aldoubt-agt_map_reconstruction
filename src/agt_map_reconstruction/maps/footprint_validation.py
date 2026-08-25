"""Offline robot-footprint validation against reconstructed navigation maps."""

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


def _as_footprint(footprint_xy_m):
    footprint = np.asarray(footprint_xy_m, dtype=float)
    if footprint.ndim != 2 or footprint.shape[1] != 2 or footprint.shape[0] < 3:
        raise ValueError('footprint_xy_m must be an Nx2 polygon with at least 3 vertices')
    return footprint


def _aisle_axis(rectangle):
    polygon = np.asarray(rectangle['polygon_xy'], dtype=float)
    if polygon.shape != (4, 2):
        raise ValueError(f"aisle {rectangle.get('aisle_id')} polygon must be 4x2")
    start = 0.5 * (polygon[0] + polygon[3])
    end = 0.5 * (polygon[1] + polygon[2])
    delta = end - start
    length_cells = float(np.linalg.norm(delta))
    if length_cells <= 0.0:
        raise ValueError(f"aisle {rectangle.get('aisle_id')} centerline has zero length")
    return start, end, delta / length_cells, length_cells


def _sample_poses(rectangle, footprint_xy_m, resolution, sample_spacing_m):
    footprint = _as_footprint(footprint_xy_m)
    if resolution <= 0.0:
        raise ValueError('resolution must be > 0')
    if sample_spacing_m <= 0.0:
        raise ValueError('sample_spacing_m must be > 0')

    start, _, unit, length_cells = _aisle_axis(rectangle)
    length_m = length_cells * float(resolution)
    yaw = math.atan2(unit[1], unit[0])

    # The aisle axis is the robot local +X direction. Respect asymmetric
    # base_link footprints by trimming each end independently.
    rear_extent_m = max(0.0, -float(np.min(footprint[:, 0])))
    front_extent_m = max(0.0, float(np.max(footprint[:, 0])))
    usable_m = length_m - rear_extent_m - front_extent_m
    if usable_m < -1e-12:
        return [], yaw

    count = max(1, int(math.ceil(max(usable_m, 0.0) / sample_spacing_m)) + 1)
    distances_m = np.linspace(rear_extent_m, length_m - front_extent_m, count)
    poses = []
    for distance_m in distances_m:
        xy = start + unit * (distance_m / resolution)
        poses.append((float(xy[0]), float(xy[1]), float(yaw)))
    return poses, yaw


def _footprint_cells(shape, footprint_xy_m, pose, resolution):
    footprint = _as_footprint(footprint_xy_m)
    x, y, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    polygon_cells = np.array([x, y]) + (footprint @ rotation.T) / float(resolution)

    h, w = shape
    out_of_bounds = bool(
        np.any(polygon_cells[:, 0] < 0.0)
        or np.any(polygon_cells[:, 0] > (w - 1))
        or np.any(polygon_cells[:, 1] < 0.0)
        or np.any(polygon_cells[:, 1] > (h - 1))
    )

    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(polygon_cells).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool), out_of_bounds


def _validate_one_aisle(base_map, rectangle, footprint_xy_m, resolution,
                         sample_spacing_m, candidate_mask, allow_unknown,
                         blocked_distance_m):
    poses, yaw = _sample_poses(rectangle, footprint_xy_m, resolution, sample_spacing_m)
    if not poses:
        return {
            'aisle_id': int(rectangle['aisle_id']),
            'label': rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}"),
            'width_m': float(rectangle.get('width_m', 0.0)),
            'length_m': float(rectangle.get('length_m', 0.0)),
            'heading_rad': float(yaw),
            'pose_count': 0,
            'passed': False,
            'collision_pose_count': 0,
            'unknown_overlap_pose_count': 0,
            'candidate_overlap_pose_count': 0,
            'out_of_bounds_pose_count': 0,
            'min_blocked_clearance_m': 0.0,
            'first_failure_reason': 'footprint_longer_than_aisle',
            'first_failure_pose': None,
        }

    occupied = base_map == OCCUPIED_VALUE
    unknown = base_map == UNKNOWN_VALUE

    collision_count = 0
    unknown_count = 0
    candidate_count = 0
    out_of_bounds_count = 0
    min_clearance = float('inf')
    first_failure_reason = None
    first_failure_pose = None

    for pose in poses:
        footprint_mask, out_of_bounds = _footprint_cells(
            base_map.shape, footprint_xy_m, pose, resolution
        )
        collision = bool(np.any(footprint_mask & occupied))
        unknown_overlap = bool(np.any(footprint_mask & unknown))
        candidate_overlap = bool(
            candidate_mask is not None and np.any(footprint_mask & candidate_mask)
        )

        pose_clearance = (
            float(np.min(blocked_distance_m[footprint_mask]))
            if np.any(footprint_mask) else 0.0
        )
        min_clearance = min(min_clearance, pose_clearance)
        collision_count += int(collision)
        unknown_count += int(unknown_overlap)
        candidate_count += int(candidate_overlap)
        out_of_bounds_count += int(out_of_bounds)

        reason = None
        if out_of_bounds:
            reason = 'out_of_bounds'
        elif collision:
            reason = 'occupied'
        elif unknown_overlap and not allow_unknown:
            reason = 'unknown'

        if reason is not None and first_failure_reason is None:
            first_failure_reason = reason
            first_failure_pose = {
                'x_cell': pose[0],
                'y_cell': pose[1],
                'yaw_rad': pose[2],
            }

    passed = (
        out_of_bounds_count == 0
        and collision_count == 0
        and (allow_unknown or unknown_count == 0)
    )

    return {
        'aisle_id': int(rectangle['aisle_id']),
        'label': rectangle.get('label', f"A{int(rectangle['aisle_id']):02d}"),
        'width_m': float(rectangle.get('width_m', 0.0)),
        'length_m': float(rectangle.get('length_m', 0.0)),
        'heading_rad': float(yaw),
        'pose_count': len(poses),
        'passed': bool(passed),
        'collision_pose_count': int(collision_count),
        'unknown_overlap_pose_count': int(unknown_count),
        'candidate_overlap_pose_count': int(candidate_count),
        'out_of_bounds_pose_count': int(out_of_bounds_count),
        'min_blocked_clearance_m': float(min_clearance),
        'first_failure_reason': first_failure_reason,
        'first_failure_pose': first_failure_pose,
    }


def validate_aisle_footprints(base_map, aisle_rectangles, footprint_xy_m,
                              resolution, sample_spacing_m=0.10,
                              candidate_mask=None, allow_unknown=False):
    """Validate a polygon robot footprint along every recovered aisle centerline.

    Arrays use the repository grid convention (y, x), with aisle polygons in
    grid-cell coordinates. Footprint vertices are in metres in the robot frame.
    Unknown space blocks validation by default. Candidate overlap is advisory.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError('base_map must be a 2D array')
    footprint = _as_footprint(footprint_xy_m)

    candidate = None
    if candidate_mask is not None:
        candidate = np.asarray(candidate_mask, dtype=bool)
        if candidate.shape != base.shape:
            raise ValueError('candidate_mask shape must match base_map')

    blocked = base == OCCUPIED_VALUE
    if not allow_unknown:
        blocked |= base == UNKNOWN_VALUE
    blocked_distance_m = ndimage.distance_transform_edt(~blocked) * float(resolution)

    aisle_results = [
        _validate_one_aisle(
            base,
            rectangle,
            footprint,
            float(resolution),
            float(sample_spacing_m),
            candidate,
            bool(allow_unknown),
            blocked_distance_m,
        )
        for rectangle in aisle_rectangles
    ]

    passed = [item for item in aisle_results if item['passed']]
    failed = [item for item in aisle_results if not item['passed']]
    return {
        'policy': {
            'allow_unknown': bool(allow_unknown),
            'sample_spacing_m': float(sample_spacing_m),
        },
        'summary': {
            'pass_count': len(passed),
            'fail_count': len(failed),
            'total_aisles': len(aisle_results),
            'failed_aisles': [item['label'] for item in failed],
        },
        'aisles': aisle_results,
    }


def write_footprint_validation_bundle(base_map, aisle_rectangles,
                                      footprint_xy_m, output_dir, resolution,
                                      sample_spacing_m=0.10,
                                      candidate_mask=None,
                                      allow_unknown=False,
                                      footprint_name='robot'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    footprint = _as_footprint(footprint_xy_m)

    result = validate_aisle_footprints(
        base_map=base_map,
        aisle_rectangles=aisle_rectangles,
        footprint_xy_m=footprint,
        resolution=resolution,
        sample_spacing_m=sample_spacing_m,
        candidate_mask=candidate_mask,
        allow_unknown=allow_unknown,
    )
    result['footprint'] = {
        'name': str(footprint_name),
        'polygon_xy_m': footprint.tolist(),
    }
    result['resolution_m'] = float(resolution)

    json_path = output / 'aisle_footprint_validation.json'
    csv_path = output / 'aisle_footprint_validation.csv'
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    fields = [
        'aisle_id', 'label', 'width_m', 'length_m', 'pose_count', 'passed',
        'collision_pose_count', 'unknown_overlap_pose_count',
        'candidate_overlap_pose_count', 'out_of_bounds_pose_count',
        'min_blocked_clearance_m',
        'first_failure_reason',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for aisle in result['aisles']:
            writer.writerow({key: aisle.get(key) for key in fields})

    return result
