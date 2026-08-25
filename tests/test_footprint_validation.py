import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def _api():
    try:
        from agt_map_reconstruction.maps.footprint_validation import (
            FREE_VALUE,
            OCCUPIED_VALUE,
            UNKNOWN_VALUE,
            validate_aisle_footprints,
            write_footprint_validation_bundle,
        )
    except ImportError as exc:
        pytest.fail(f'EXP004 footprint validation API is missing: {exc}')
    return FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE, validate_aisle_footprints, write_footprint_validation_bundle


def _rect(x0, y0, x1, y1, aisle_id=1):
    return {
        'aisle_id': aisle_id,
        'label': f'A{aisle_id:02d}',
        'polygon_xy': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        'width_m': (y1 - y0) * 0.1,
        'length_m': (x1 - x0) * 0.1,
    }


def _footprint():
    # base_link-centred 0.6 m x 0.4 m rectangle
    return np.array([
        [0.30, 0.20],
        [0.30, -0.20],
        [-0.30, -0.20],
        [-0.30, 0.20],
    ], dtype=float)


def test_polygon_footprint_passes_clear_aisle():
    FREE, OCC, UNKNOWN, validate, _ = _api()
    grid = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    grid[15:36, 10:91] = FREE

    result = validate(
        grid,
        [_rect(10, 15, 90, 35)],
        _footprint(),
        resolution=0.1,
        sample_spacing_m=0.5,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is True
    assert aisle['collision_pose_count'] == 0
    assert aisle['unknown_overlap_pose_count'] == 0
    assert aisle['out_of_bounds_pose_count'] == 0
    assert result['summary']['pass_count'] == 1


def test_static_obstacle_collision_fails_aisle():
    FREE, OCC, UNKNOWN, validate, _ = _api()
    grid = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    grid[15:36, 10:91] = FREE
    grid[23:28, 49:52] = OCC

    result = validate(
        grid,
        [_rect(10, 15, 90, 35)],
        _footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is False
    assert aisle['collision_pose_count'] > 0
    assert aisle['first_failure_reason'] == 'occupied'


def test_unknown_is_blocking_by_default_but_can_be_allowed():
    FREE, OCC, UNKNOWN, validate, _ = _api()
    grid = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    grid[15:36, 10:91] = FREE
    grid[23:28, 49:52] = UNKNOWN

    blocked = validate(
        grid,
        [_rect(10, 15, 90, 35)],
        _footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
    )
    allowed = validate(
        grid,
        [_rect(10, 15, 90, 35)],
        _footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
        allow_unknown=True,
    )

    assert blocked['aisles'][0]['passed'] is False
    assert blocked['aisles'][0]['unknown_overlap_pose_count'] > 0
    assert blocked['aisles'][0]['first_failure_reason'] == 'unknown'
    assert allowed['aisles'][0]['passed'] is True
    assert allowed['aisles'][0]['unknown_overlap_pose_count'] > 0


def test_candidate_overlap_is_reported_but_advisory():
    FREE, OCC, UNKNOWN, validate, _ = _api()
    grid = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    grid[15:36, 10:91] = FREE
    candidate = np.zeros_like(grid, dtype=bool)
    candidate[23:28, 49:52] = True

    result = validate(
        grid,
        [_rect(10, 15, 90, 35)],
        _footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
        candidate_mask=candidate,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is True
    assert aisle['candidate_overlap_pose_count'] > 0


def test_bundle_writes_json_and_csv(tmp_path):
    FREE, OCC, UNKNOWN, _, write_bundle = _api()
    grid = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    grid[15:36, 10:91] = FREE

    result = write_bundle(
        base_map=grid,
        aisle_rectangles=[_rect(10, 15, 90, 35)],
        footprint_xy_m=_footprint(),
        output_dir=tmp_path,
        resolution=0.1,
        sample_spacing_m=0.5,
        footprint_name='test_robot',
    )

    assert (tmp_path / 'aisle_footprint_validation.json').exists()
    assert (tmp_path / 'aisle_footprint_validation.csv').exists()
    payload = json.loads((tmp_path / 'aisle_footprint_validation.json').read_text())
    assert payload['footprint']['name'] == 'test_robot'
    assert payload['summary']['pass_count'] == 1
    assert result['summary']['pass_count'] == 1


def test_validate_robot_footprint_cli_reads_exp003_bundle(tmp_path):
    import os
    import subprocess
    import yaml
    import cv2

    FREE, OCC, UNKNOWN, _, _ = _api()
    internal = np.full((50, 100), UNKNOWN, dtype=np.uint8)
    internal[15:36, 10:91] = FREE
    pgm_path = tmp_path / 'navigation_base_map.pgm'
    cv2.imwrite(str(pgm_path), np.flipud(internal))

    yaml_path = tmp_path / 'navigation_base_map.yaml'
    yaml_path.write_text(yaml.safe_dump({
        'image': pgm_path.name,
        'mode': 'trinary',
        'resolution': 0.1,
        'origin': [0.0, 0.0, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }))

    aisles_path = tmp_path / 'aisle_rectangles.json'
    aisles_path.write_text(json.dumps({'rectangles': [_rect(10, 15, 90, 35)]}))
    footprint_path = tmp_path / 'footprint.json'
    footprint_path.write_text(json.dumps({
        'name': 'test_robot',
        'polygon_xy_m': _footprint().tolist(),
    }))
    candidate_path = tmp_path / 'candidate_mask.npy'
    np.save(candidate_path, np.zeros_like(internal, dtype=np.uint8))

    output = tmp_path / 'out'
    script = Path(__file__).resolve().parents[1] / 'tools' / 'validate_robot_footprint.py'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    completed = subprocess.run(
        [
            sys.executable, str(script),
            '--map-pgm', str(pgm_path),
            '--map-yaml', str(yaml_path),
            '--aisles', str(aisles_path),
            '--footprint', str(footprint_path),
            '--candidate-mask', str(candidate_path),
            '--output', str(output),
            '--sample-spacing', '0.5',
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / 'aisle_footprint_validation.json').exists()
    payload = json.loads((output / 'aisle_footprint_validation.json').read_text())
    assert payload['summary']['pass_count'] == 1
    assert 'passed aisles: 1/1' in completed.stdout
