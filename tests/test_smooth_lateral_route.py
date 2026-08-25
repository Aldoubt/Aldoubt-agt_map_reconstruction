import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

FREE = np.uint8(254)
UNKNOWN = np.uint8(205)
OCCUPIED = np.uint8(0)


def _api():
    try:
        from agt_map_reconstruction.maps.smooth_lateral_route import (
            search_smooth_lateral_routes,
            write_smooth_route_bundle,
        )
    except ImportError as exc:
        pytest.fail(f'EXP004-B2 smooth route API is missing: {exc}')
    return search_smooth_lateral_routes, write_smooth_route_bundle


def _rect(x0=10, y0=20, x1=110, y1=40, aisle_id=1, resolution=0.1):
    return {
        'aisle_id': aisle_id,
        'label': f'A{aisle_id:02d}',
        'polygon_xy': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        'width_m': (y1 - y0) * resolution,
        'length_m': (x1 - x0) * resolution,
    }


def _footprint():
    return np.array([
        [0.30, 0.20],
        [0.30, -0.20],
        [-0.30, -0.20],
        [-0.30, 0.20],
    ], dtype=float)


def _grid():
    grid = np.full((60, 120), UNKNOWN, dtype=np.uint8)
    grid[20:41, 10:111] = FREE
    return grid


def test_smooth_route_recovers_when_no_constant_offset_can_pass():
    search, _ = _api()
    grid = _grid()
    # Lower-half obstacle first -> go +Y; upper-half obstacle later -> go -Y.
    grid[20:30, 42:48] = OCCUPIED
    grid[31:41, 72:78] = OCCUPIED

    baseline = {'aisles': [{'label': 'A01', 'passed': False}]}
    result = search(
        grid, [_rect()], _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
        baseline_b1=baseline,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is True
    assert aisle['route_recovered_from_b1'] is True
    offsets = np.asarray(aisle['control_offsets_m'])
    assert offsets.max() > 0.2
    assert offsets.min() < -0.2
    assert aisle['blocking_pose_count'] == 0


def test_smooth_route_respects_max_offset_change():
    search, _ = _api()
    grid = _grid()
    grid[20:30, 42:48] = OCCUPIED
    grid[31:41, 72:78] = OCCUPIED

    result = search(
        grid, [_rect()], _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
    )
    offsets = np.asarray(result['aisles'][0]['control_offsets_m'])
    assert np.max(np.abs(np.diff(offsets))) <= 0.2 + 1e-9
    assert result['aisles'][0]['max_offset_step_m'] <= 0.2 + 1e-9


def test_unknown_remains_blocking_by_default():
    search, _ = _api()
    grid = _grid()
    # Full-width unknown strip cannot be bypassed.
    grid[20:41, 58:62] = UNKNOWN

    result = search(
        grid, [_rect()], _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
    )
    assert result['aisles'][0]['passed'] is False

    diagnostic = search(
        grid, [_rect()], _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
        allow_unknown=True,
    )
    assert diagnostic['aisles'][0]['passed'] is True


def test_clear_straight_aisle_prefers_small_offset_and_heading_change():
    search, _ = _api()
    result = search(
        _grid(), [_rect()], _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
    )
    aisle = result['aisles'][0]
    assert aisle['passed'] is True
    assert max(abs(v) for v in aisle['control_offsets_m']) <= 0.1
    assert aisle['max_heading_deviation_rad'] < 1e-6


def test_bundle_writes_json_csv_and_focus_overlays(tmp_path):
    _, write_bundle = _api()
    result = write_bundle(
        base_map=_grid(), aisle_rectangles=[_rect()],
        footprint_xy_m=_footprint(), output_dir=tmp_path,
        resolution=0.1, sample_spacing_m=0.1,
        control_spacing_m=0.5, offset_step_m=0.1,
        max_offset_change_m=0.2, footprint_name='test_robot',
        focus_aisles=['A01'],
    )
    expected = {
        'smooth_route_search.json', 'smooth_route_search.csv',
        'smooth_route_overlay.png', 'A01_smooth_route_overlay.png',
    }
    assert expected.issubset({p.name for p in tmp_path.iterdir()})
    payload = json.loads((tmp_path / 'smooth_route_search.json').read_text())
    assert payload['summary']['pass_count'] == 1
    assert payload['footprint']['name'] == 'test_robot'
    image = cv2.imread(str(tmp_path / 'smooth_route_overlay.png'))
    assert image is not None


def test_summary_compares_against_b1_baseline():
    search, _ = _api()
    baseline = {
        'aisles': [
            {'label': 'A01', 'passed': False},
            {'label': 'A02', 'passed': True},
        ]
    }
    result = search(
        _grid(), [_rect(aisle_id=1), _rect(y0=20, y1=40, aisle_id=2)],
        _footprint(), resolution=0.1,
        sample_spacing_m=0.1, control_spacing_m=0.5,
        offset_step_m=0.1, max_offset_change_m=0.2,
        baseline_b1=baseline,
    )
    assert result['summary']['baseline_b1_pass_count'] == 1
    assert result['summary']['pass_count'] == 2
    assert result['summary']['recovered_from_b1_count'] == 1
    assert result['summary']['recovered_from_b1_aisles'] == ['A01']


def test_cli_reads_exp003_assets_and_b1_baseline(tmp_path):
    import os
    import subprocess
    import yaml

    internal = _grid()
    pgm = tmp_path / 'navigation_base_map.pgm'
    cv2.imwrite(str(pgm), np.flipud(internal))
    map_yaml = tmp_path / 'navigation_base_map.yaml'
    map_yaml.write_text(yaml.safe_dump({
        'image': pgm.name, 'mode': 'trinary', 'resolution': 0.1,
        'origin': [0.0, 0.0, 0.0], 'negate': 0,
        'occupied_thresh': 0.65, 'free_thresh': 0.196,
    }))
    aisles = tmp_path / 'aisles.json'
    aisles.write_text(json.dumps({'rectangles': [_rect()]}))
    footprint = tmp_path / 'footprint.json'
    footprint.write_text(json.dumps({'name': 'test_robot', 'polygon_xy_m': _footprint().tolist()}))
    candidate = tmp_path / 'candidate.npy'
    np.save(candidate, np.zeros_like(internal, dtype=np.uint8))
    baseline = tmp_path / 'b1.json'
    baseline.write_text(json.dumps({'aisles': [{'label': 'A01', 'passed': True, 'best_offset_m': 0.0}]}))

    script = Path(__file__).resolve().parents[1] / 'tools' / 'search_smooth_lateral_routes.py'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    completed = subprocess.run([
        sys.executable, str(script),
        '--map-pgm', str(pgm), '--map-yaml', str(map_yaml),
        '--aisles', str(aisles), '--footprint', str(footprint),
        '--candidate-mask', str(candidate), '--baseline-b1', str(baseline),
        '--output', str(tmp_path / 'out'), '--focus-aisles', 'A01',
    ], env=env, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    assert 'smooth routes passed: 1/1' in completed.stdout
    assert 'baseline B1 passed: 1/1' in completed.stdout
    assert (tmp_path / 'out' / 'A01_smooth_route_overlay.png').exists()
