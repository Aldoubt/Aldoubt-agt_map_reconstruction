#!/usr/bin/env python3
"""Run EXP004-B2 smooth lateral route search on EXP003/EXP004 assets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from agt_map_reconstruction.maps.smooth_lateral_route import write_smooth_route_bundle
from agt_map_reconstruction.maps.review_corrections import load_review


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


def _load_json_optional(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(
        description='Search smooth spatially varying lateral routes inside recovered aisles.'
    )
    parser.add_argument('--map-pgm', required=True)
    parser.add_argument('--map-yaml', required=True)
    parser.add_argument('--aisles', required=True)
    parser.add_argument('--footprint', required=True)
    parser.add_argument('--candidate-mask')
    parser.add_argument('--baseline-b1', help='EXP004-B1 aisle_offset_search.json for comparison')
    parser.add_argument('--output', default='results/EXP004/smooth-lateral-route-v1')
    parser.add_argument('--sample-spacing', type=float, default=0.10)
    parser.add_argument('--control-spacing', type=float, default=0.50)
    parser.add_argument('--offset-step', type=float, default=0.05)
    parser.add_argument('--max-offset-change', type=float, default=0.10)
    parser.add_argument(
        '--endpoint-trim', type=float, default=0.0,
        help='diagnostic handoff trim at both aisle ends; 0.0 keeps strict full-length validation',
    )
    parser.add_argument('--focus-aisles', nargs='*', default=['A05', 'A07', 'A20'])
    parser.add_argument('--allow-unknown', action='store_true')
    parser.add_argument('--review', help='optional manual route review JSON')
    args = parser.parse_args()

    map_yaml = yaml.safe_load(Path(args.map_yaml).read_text(encoding='utf-8'))
    resolution = float(map_yaml['resolution'])
    base_map = _load_map(args.map_pgm)
    aisles = _load_aisles(args.aisles)
    footprint_name, footprint = _load_footprint(args.footprint)
    candidate = np.load(args.candidate_mask).astype(bool) if args.candidate_mask else None
    baseline = _load_json_optional(args.baseline_b1)

    result = write_smooth_route_bundle(
        base_map=base_map,
        aisle_rectangles=aisles,
        footprint_xy_m=footprint,
        output_dir=args.output,
        resolution=resolution,
        sample_spacing_m=args.sample_spacing,
        control_spacing_m=args.control_spacing,
        offset_step_m=args.offset_step,
        max_offset_change_m=args.max_offset_change,
        endpoint_trim_m=args.endpoint_trim,
        candidate_mask=candidate,
        allow_unknown=args.allow_unknown,
        baseline_b1=baseline,
        footprint_name=footprint_name,
        focus_aisles=args.focus_aisles,
        manual_review=load_review(args.review) if args.review else None,
    )

    summary = result['summary']
    total = summary['total_aisles']
    print('output:', Path(args.output))
    print('footprint:', footprint_name)
    print('resolution:', resolution)
    print('allow_unknown:', args.allow_unknown)
    print('endpoint_trim_m:', args.endpoint_trim)
    if args.endpoint_trim > 0.0:
        print('mode: diagnostic handoff trim (not strict full-length acceptance)')
    if summary.get('baseline_b1_available'):
        print(f"baseline B1 passed: {summary['baseline_b1_pass_count']}/{total}")
    print(f"smooth routes passed: {summary['pass_count']}/{total}")
    if summary.get('baseline_b1_available'):
        print('recovered from B1:', summary['recovered_from_b1_count'])
        if summary['recovered_from_b1_aisles']:
            print('recovered aisles:', ', '.join(summary['recovered_from_b1_aisles']))
    if summary['failed_aisles']:
        print('failed aisles:', ', '.join(summary['failed_aisles']))

    regions = {'entry': [], 'interior': [], 'exit': [], 'aisle_geometry': []}
    for item in result['aisles']:
        if not item['passed'] and item.get('failure_region') in regions:
            regions[item['failure_region']].append(item['label'])
    for region, labels in regions.items():
        if labels:
            print(f'{region} failures:', ', '.join(labels))


if __name__ == '__main__':
    main()
