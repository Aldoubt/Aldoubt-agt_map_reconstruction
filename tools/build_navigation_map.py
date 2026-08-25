#!/usr/bin/env python3
"""Build navigation-map-v2 artifacts from semantic reconstruction outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from agt_map_reconstruction.maps.navigation_export import write_navigation_bundle


def _load_aisles(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        rectangles = payload.get('rectangles')
    else:
        rectangles = payload
    if not isinstance(rectangles, list):
        raise ValueError('aisle JSON must be a list or contain a rectangles list')
    return rectangles


def main():
    parser = argparse.ArgumentParser(
        description='Export a navigation-oriented static base map and clearance validation report.'
    )
    parser.add_argument('--semantic-labels', required=True, help='semantic_labels.npy')
    parser.add_argument('--aisles', required=True, help='aisle_rectangles.json')
    parser.add_argument('--output', default='results/EXP003/navigation-map-v2')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--origin', type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        '--clearance-radii',
        type=float,
        nargs='+',
        default=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
        help='robot-equivalent safety radii in metres',
    )
    args = parser.parse_args()

    semantic = np.load(args.semantic_labels)
    aisles = _load_aisles(args.aisles)
    bundle = write_navigation_bundle(
        semantic_labels=semantic,
        aisle_rectangles=aisles,
        output_dir=args.output,
        resolution=args.resolution,
        origin=args.origin,
        clearance_radii_m=args.clearance_radii,
    )

    validation = bundle['validation']
    print('output:', Path(args.output))
    print('map_server_yaml_valid:', validation['map_server_yaml_valid'])
    print('gray_semantics_valid:', validation['gray_semantics_valid'])
    print('candidate_cells:', validation['candidate_cell_count'])
    print('pillar_cells:', validation['pillar_cell_count'])
    print('pillar_as_free_cells:', validation['pillar_as_free_cell_count'])
    print('static_obstacle_semantics_valid:', validation['static_obstacle_semantics_valid'])
    for key, metrics in validation['clearance_tests'].items():
        print(
            f"clearance {key} m: "
            f"{metrics['pass_count']}/{metrics['total_aisles']} aisles pass"
        )


if __name__ == '__main__':
    main()
