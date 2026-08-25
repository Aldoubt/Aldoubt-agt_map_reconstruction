import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def _api():
    try:
        from agt_map_reconstruction.maps.in_aisle_route_search import (
            lateral_offset_candidates,
            search_constant_offset_routes,
            write_offset_search_bundle,
        )
    except ImportError as exc:
        pytest.fail(f'EXP004-B in-aisle route search API is missing: {exc}')
    return lateral_offset_candidates, search_constant_offset_routes, write_offset_search_bundle


def _rect(x0, y0, x1, y1, aisle_id=1, resolution=0.1):
    return {
        'aisle_id': aisle_id,
        'label': f'A{aisle_id:02d}',
        'polygon_xy': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        'width_m': (y1 - y0) * resolution,
        'length_m': (x1 - x0) * resolution,
    }


def _footprint():
    # 0.60 m long x 0.40 m wide, centred at base_link.
    return np.array([
        [0.30, 0.20],
        [0.30, -0.20],
        [-0.30, -0.20],
        [-0.30, 0.20],
    ], dtype=float)


def _clear_grid():
    from agt_map_reconstruction.maps.footprint_validation import FREE_VALUE, UNKNOWN_VALUE
    grid = np.full((60, 120), UNKNOWN_VALUE, dtype=np.uint8)
    grid[20:41, 10:111] = FREE_VALUE
    return grid


def test_offset_candidates_respect_aisle_and_footprint_width():
    candidates_fn, _, _ = _api()
    rectangle = _rect(10, 20, 110, 40)

    offsets = candidates_fn(
        rectangle,
        _footprint(),
        resolution=0.1,
        offset_step_m=0.20,
    )

    assert offsets[0] == pytest.approx(-0.80)
    assert offsets[-1] == pytest.approx(0.80)
    assert any(abs(value) < 1e-12 for value in offsets)
    assert all(-0.80 - 1e-9 <= value <= 0.80 + 1e-9 for value in offsets)


def test_constant_offset_search_recovers_route_when_centerline_hits_obstacle():
    _, search, _ = _api()
    from agt_map_reconstruction.maps.footprint_validation import OCCUPIED_VALUE

    grid = _clear_grid()
    grid[28:33, 52:58] = OCCUPIED_VALUE

    result = search(
        base_map=grid,
        aisle_rectangles=[_rect(10, 20, 110, 40)],
        footprint_xy_m=_footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
        offset_step_m=0.10,
    )

    aisle = result['aisles'][0]
    assert aisle['centerline_passed'] is False
    assert aisle['passed'] is True
    assert aisle['route_recovered'] is True
    assert aisle['feasible_offset_count'] > 0
    assert abs(aisle['best_offset_m']) >= 0.30
    assert aisle['best_min_blocked_clearance_m'] > 0.0


def test_constant_offset_search_reports_no_route_when_obstacle_spans_aisle():
    _, search, _ = _api()
    from agt_map_reconstruction.maps.footprint_validation import OCCUPIED_VALUE

    grid = _clear_grid()
    grid[20:41, 52:58] = OCCUPIED_VALUE

    result = search(
        base_map=grid,
        aisle_rectangles=[_rect(10, 20, 110, 40)],
        footprint_xy_m=_footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
        offset_step_m=0.10,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is False
    assert aisle['feasible_offset_count'] == 0
    assert aisle['best_offset_m'] is None
    assert aisle['best_attempt_offset_m'] is not None
    assert aisle['best_attempt_blocking_pose_count'] > 0


def test_best_route_prefers_clearance_away_from_side_obstacle():
    _, search, _ = _api()
    from agt_map_reconstruction.maps.footprint_validation import OCCUPIED_VALUE

    grid = _clear_grid()
    # Persistent obstacle band on the +Y side. The safest constant route should
    # move toward negative lateral offsets while remaining inside the aisle.
    grid[34:39, 30:91] = OCCUPIED_VALUE

    result = search(
        base_map=grid,
        aisle_rectangles=[_rect(10, 20, 110, 40)],
        footprint_xy_m=_footprint(),
        resolution=0.1,
        sample_spacing_m=0.2,
        offset_step_m=0.10,
    )

    aisle = result['aisles'][0]
    assert aisle['passed'] is True
    assert aisle['best_offset_m'] < 0.0
    assert aisle['best_min_blocked_clearance_m'] > 0.0


def test_bundle_writes_json_csv_and_route_overlays(tmp_path):
    _, _, write_bundle = _api()
    grid = _clear_grid()

    result = write_bundle(
        base_map=grid,
        aisle_rectangles=[_rect(10, 20, 110, 40)],
        footprint_xy_m=_footprint(),
        output_dir=tmp_path,
        resolution=0.1,
        sample_spacing_m=0.2,
        offset_step_m=0.10,
        footprint_name='test_robot',
        focus_aisle='A01',
    )

    expected = {
        'aisle_offset_search.json',
        'aisle_offset_search.csv',
        'route_overlay.png',
        'A01_route_overlay.png',
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    payload = json.loads((tmp_path / 'aisle_offset_search.json').read_text())
    assert payload['footprint']['name'] == 'test_robot'
    assert payload['summary']['pass_count'] == 1
    assert payload['aisles'][0]['best_offset_m'] is not None
    assert result['summary']['pass_count'] == 1
    assert cv2.imread(str(tmp_path / 'route_overlay.png')) is not None


def test_search_cli_reads_exp003_assets_and_recovers_offset_route(tmp_path):
    import os
    import subprocess
    import yaml

    from agt_map_reconstruction.maps.footprint_validation import (
        FREE_VALUE,
        OCCUPIED_VALUE,
        UNKNOWN_VALUE,
    )

    internal = np.full((60, 120), UNKNOWN_VALUE, dtype=np.uint8)
    internal[20:41, 10:111] = FREE_VALUE
    internal[28:33, 52:58] = OCCUPIED_VALUE

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
    aisles_path.write_text(json.dumps({'rectangles': [_rect(10, 20, 110, 40)]}))
    footprint_path = tmp_path / 'footprint.json'
    footprint_path.write_text(json.dumps({
        'name': 'test_robot',
        'polygon_xy_m': _footprint().tolist(),
    }))
    candidate_path = tmp_path / 'candidate_mask.npy'
    np.save(candidate_path, np.zeros_like(internal, dtype=np.uint8))

    output = tmp_path / 'out'
    script = Path(__file__).resolve().parents[1] / 'tools' / 'search_in_aisle_offsets.py'
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
            '--sample-spacing', '0.2',
            '--offset-step', '0.1',
            '--focus-aisle', 'A01',
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / 'aisle_offset_search.json').read_text())
    assert payload['summary']['pass_count'] == 1
    assert payload['aisles'][0]['route_recovered'] is True
    assert 'recovered routes: 1' in completed.stdout
    assert (output / 'A01_route_overlay.png').exists()
