import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    build_navigation_layers,
    build_map_yaml,
    validate_navigation_map,
)


def _rect(x0, y0, x1, y1, aisle_id=1, width_m=1.0, length_m=4.0):
    return {
        'aisle_id': aisle_id,
        'label': f'A{aisle_id:02d}',
        'polygon_xy': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        'width_m': width_m,
        'length_m': length_m,
    }


def test_base_map_recovers_aisle_prior_without_erasing_hard_obstacles():
    semantic = np.zeros((7, 10), dtype=np.uint8)
    semantic[2:5, 1:9] = 1
    semantic[3, 4] = 5
    semantic[3, 6] = 2

    layers = build_navigation_layers(semantic, [_rect(1, 2, 8, 4)])

    assert layers.base_map[3, 4] == FREE_VALUE
    assert bool(layers.candidate_mask[3, 4]) is True
    assert layers.base_map[3, 6] == OCCUPIED_VALUE
    assert layers.base_map[0, 0] == UNKNOWN_VALUE
    assert layers.base_map[2, 2] == FREE_VALUE


def test_map_yaml_uses_valid_nav2_trinary_threshold_order():
    config = build_map_yaml('navigation_base_map.pgm', resolution=0.05, origin=(1.0, 2.0, 0.0))

    assert config['mode'] == 'trinary'
    assert config['resolution'] == 0.05
    assert config['origin'] == [1.0, 2.0, 0.0]
    assert config['free_thresh'] < config['occupied_thresh']
    assert config['free_thresh'] == 0.196
    assert config['occupied_thresh'] == 0.65


def test_clearance_validation_reports_pass_and_fail_by_robot_radius():
    semantic = np.zeros((30, 100), dtype=np.uint8)
    semantic[5:25, 5:95] = 1
    rectangle = _rect(5, 5, 94, 24, width_m=1.0, length_m=4.5)
    layers = build_navigation_layers(semantic, [rectangle])

    result = validate_navigation_map(
        layers.base_map,
        [rectangle],
        resolution=0.05,
        clearance_radii_m=(0.30, 0.55),
    )

    assert result['clearance_tests']['0.30']['pass_count'] == 1
    assert result['clearance_tests']['0.30']['total_aisles'] == 1
    assert result['clearance_tests']['0.55']['pass_count'] == 0
    assert result['aisles'][0]['clearance_pass']['0.30'] is True
    assert result['aisles'][0]['clearance_pass']['0.55'] is False


def test_validation_rejects_noncanonical_gray_values():
    base = np.full((4, 4), FREE_VALUE, dtype=np.uint8)
    base[0, 0] = 127

    result = validate_navigation_map(base, [], resolution=0.05, clearance_radii_m=(0.2,))

    assert result['gray_semantics_valid'] is False
    assert result['unexpected_gray_values'] == [127]


def test_write_navigation_bundle_creates_nav2_and_validation_artifacts(tmp_path):
    from agt_map_reconstruction.maps.navigation_export import write_navigation_bundle

    semantic = np.zeros((6, 10), dtype=np.uint8)
    semantic[2:5, 1:9] = 1
    semantic[3, 4] = 6
    rectangles = [_rect(1, 2, 8, 4, width_m=0.75, length_m=0.35)]

    result = write_navigation_bundle(
        semantic_labels=semantic,
        aisle_rectangles=rectangles,
        output_dir=tmp_path,
        resolution=0.05,
        origin=(0.0, 0.0, 0.0),
        clearance_radii_m=(0.10,),
    )

    expected = {
        'navigation_base_map.pgm',
        'navigation_base_map.yaml',
        'candidate_mask.npy',
        'validation.json',
    }
    assert expected.issubset({p.name for p in tmp_path.iterdir()})
    assert result['validation']['map_server_yaml_valid'] is True
    assert result['validation']['candidate_cell_count'] == 1

    with open(tmp_path / 'navigation_base_map.pgm', 'rb') as stream:
        header = stream.readline().strip()
    assert header == b'P5'

    payload = json.loads((tmp_path / 'validation.json').read_text())
    assert payload['map_server_yaml_valid'] is True
    assert payload['candidate_cell_count'] == 1


def test_build_navigation_map_cli_writes_bundle(tmp_path):
    semantic = np.zeros((8, 12), dtype=np.uint8)
    semantic[2:6, 1:11] = 1
    semantic_path = tmp_path / 'semantic_labels.npy'
    np.save(semantic_path, semantic)

    aisles_path = tmp_path / 'aisle_rectangles.json'
    aisles_path.write_text(json.dumps({'rectangles': [_rect(1, 2, 10, 5)]}))
    output = tmp_path / 'out'

    script = Path(__file__).resolve().parents[1] / 'tools' / 'build_navigation_map.py'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    completed = subprocess.run(
        [
            sys.executable, str(script),
            '--semantic-labels', str(semantic_path),
            '--aisles', str(aisles_path),
            '--output', str(output),
            '--resolution', '0.05',
            '--clearance-radii', '0.10', '0.20',
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / 'navigation_base_map.yaml').exists()
    assert (output / 'validation.json').exists()
    assert 'clearance 0.10 m:' in completed.stdout
