#!/usr/bin/env python3
"""Run EXP004-A polygon footprint validation on EXP003 navigation-map assets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from agt_map_reconstruction.maps.footprint_validation import write_footprint_validation_bundle


def _load_aisles(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    rectangles = payload.get('rectangles') if isinstance(payload, dict) else payload
    if not isinstance(rectangles, list):
        raise ValueError('aisle JSON must be a list or contain a rectangles list')
    return rectangles


def _load_footprint(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    polygon = payload.get('polygon_xy_m')
    if polygon is None:
        raise ValueError('footprint JSON must contain polygon_xy_m')
    return payload.get('name', 'robot'), np.asarray(polygon, dtype=float)


def _load_map(pgm_path):
    image = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'failed to read map PGM: {pgm_path}')
    # map-server image rows run top-to-bottom; repository grid arrays use y-up.
    return np.flipud(image)


def main():
    parser = argparse.ArgumentParser(
        description='Validate an actual polygon robot footprint along recovered aisle centerlines.'
    )
    parser.add_argument('--map-pgm', required=True, help='EXP003 navigation_base_map.pgm')
    parser.add_argument('--map-yaml', required=True, help='EXP003 navigation_base_map.yaml')
    parser.add_argument('--aisles', required=True, help='aisle_rectangles.json')
    parser.add_argument('--footprint', required=True, help='JSON with name and polygon_xy_m')
    parser.add_argument('--candidate-mask', help='optional EXP003 candidate_mask.npy')
    parser.add_argument('--output', default='results/EXP004/robot-footprint-v1')
    parser.add_argument('--sample-spacing', type=float, default=0.10)
    parser.add_argument('--allow-unknown', action='store_true')
    args = parser.parse_args()

    map_yaml = yaml.safe_load(Path(args.map_yaml).read_text(encoding='utf-8'))
    resolution = float(map_yaml['resolution'])
    base_map = _load_map(args.map_pgm)
    aisles = _load_aisles(args.aisles)
    footprint_name, footprint = _load_footprint(args.footprint)
    candidate = np.load(args.candidate_mask).astype(bool) if args.candidate_mask else None

    result = write_footprint_validation_bundle(
        base_map=base_map,
        aisle_rectangles=aisles,
        footprint_xy_m=footprint,
        output_dir=args.output,
        resolution=resolution,
        sample_spacing_m=args.sample_spacing,
        candidate_mask=candidate,
        allow_unknown=args.allow_unknown,
        footprint_name=footprint_name,
    )

    summary = result['summary']
    print('output:', Path(args.output))
    print('footprint:', footprint_name)
    print('resolution:', resolution)
    print('allow_unknown:', args.allow_unknown)
    print(f"passed aisles: {summary['pass_count']}/{summary['total_aisles']}")
    if summary['failed_aisles']:
        print('failed aisles:', ', '.join(summary['failed_aisles']))


if __name__ == '__main__':
    main()
