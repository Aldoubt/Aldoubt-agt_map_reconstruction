#!/usr/bin/env python3
"""Run EXP004-B constant lateral-offset search on EXP003 navigation assets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from agt_map_reconstruction.maps.in_aisle_route_search import write_offset_search_bundle


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


def _load_map(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'failed to read map PGM: {path}')
    return np.flipud(image)


def main():
    parser = argparse.ArgumentParser(
        description='Search constant lateral-offset footprint routes inside recovered aisles.'
    )
    parser.add_argument('--map-pgm', required=True)
    parser.add_argument('--map-yaml', required=True)
    parser.add_argument('--aisles', required=True)
    parser.add_argument('--footprint', required=True)
    parser.add_argument('--candidate-mask')
    parser.add_argument('--output', default='results/EXP004/in-aisle-route-search-v1')
    parser.add_argument('--sample-spacing', type=float, default=0.10)
    parser.add_argument('--offset-step', type=float, default=0.05)
    parser.add_argument('--focus-aisle', default='A05')
    parser.add_argument('--allow-unknown', action='store_true')
    args = parser.parse_args()

    map_yaml = yaml.safe_load(Path(args.map_yaml).read_text(encoding='utf-8'))
    resolution = float(map_yaml['resolution'])
    base_map = _load_map(args.map_pgm)
    aisles = _load_aisles(args.aisles)
    footprint_name, footprint = _load_footprint(args.footprint)
    candidate = np.load(args.candidate_mask).astype(bool) if args.candidate_mask else None

    result = write_offset_search_bundle(
        base_map=base_map,
        aisle_rectangles=aisles,
        footprint_xy_m=footprint,
        output_dir=args.output,
        resolution=resolution,
        sample_spacing_m=args.sample_spacing,
        offset_step_m=args.offset_step,
        candidate_mask=candidate,
        allow_unknown=args.allow_unknown,
        footprint_name=footprint_name,
        focus_aisle=args.focus_aisle,
    )

    summary = result['summary']
    print('output:', Path(args.output))
    print('footprint:', footprint_name)
    print('resolution:', resolution)
    print('allow_unknown:', args.allow_unknown)
    print(f"centerline passed: {summary['centerline_pass_count']}/{summary['total_aisles']}")
    print(f"offset routes passed: {summary['pass_count']}/{summary['total_aisles']}")
    print('recovered routes:', summary['recovered_route_count'])
    if summary['recovered_aisles']:
        print('recovered aisles:', ', '.join(summary['recovered_aisles']))
    if summary['failed_aisles']:
        print('failed aisles:', ', '.join(summary['failed_aisles']))

    if args.focus_aisle:
        focus = next((item for item in result['aisles'] if item['label'] == args.focus_aisle), None)
        if focus is not None:
            print(
                f"{args.focus_aisle}: passed={focus['passed']} "
                f"best_offset_m={focus['best_offset_m']} "
                f"best_attempt_offset_m={focus['best_attempt_offset_m']}"
            )


if __name__ == '__main__':
    main()
