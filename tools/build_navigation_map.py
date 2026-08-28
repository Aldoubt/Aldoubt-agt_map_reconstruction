#!/usr/bin/env python3
"""Build navigation-map artifacts from semantic reconstruction outputs."""

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


def _load_optional_mask(path, name, shape):
    if not path:
        return None
    mask = np.load(path, allow_pickle=False).astype(bool, copy=False)
    if mask.shape != shape:
        raise ValueError(f'{name} must match semantic_labels shape')
    return mask


def main():
    parser = argparse.ArgumentParser(
        description='Export a navigation-oriented static base map and clearance validation report.'
    )
    parser.add_argument('--semantic-labels', required=True, help='semantic_labels.npy')
    parser.add_argument('--aisles', required=True, help='aisle_rectangles.json')
    parser.add_argument('--trusted-free-mask', help='explicit evidence-approved free-space mask (.npy)')
    parser.add_argument('--uncertainty-mask', help='cells that must remain unknown unless hard occupied (.npy)')
    parser.add_argument(
        '--no-promote-aisle-prior',
        action='store_true',
        help='do not turn aisle geometry into free space without explicit evidence',
    )
    parser.add_argument(
        '--promote-candidates-in-aisles',
        action='store_true',
        help='legacy diagnostic option; candidate cells remain advisory by default',
    )
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

    semantic = np.load(args.semantic_labels, allow_pickle=False)
    aisles = _load_aisles(args.aisles)
    trusted = _load_optional_mask(
        args.trusted_free_mask,
        'trusted_free_mask',
        semantic.shape,
    )
    uncertainty = _load_optional_mask(
        args.uncertainty_mask,
        'uncertainty_mask',
        semantic.shape,
    )
    bundle = write_navigation_bundle(
        semantic_labels=semantic,
        aisle_rectangles=aisles,
        output_dir=args.output,
        resolution=args.resolution,
        origin=args.origin,
        clearance_radii_m=args.clearance_radii,
        promote_aisle_prior=not args.no_promote_aisle_prior,
        promote_candidates_in_aisles=args.promote_candidates_in_aisles,
        trusted_free_mask=trusted,
        uncertainty_mask=uncertainty,
    )

    validation = bundle['validation']
    print('output:', Path(args.output))
    print('map_server_yaml_valid:', validation['map_server_yaml_valid'])
    print('gray_semantics_valid:', validation['gray_semantics_valid'])
    print('aisle_prior_promotion_enabled:', validation['aisle_prior_promotion_enabled'])
    print('candidate_cells:', validation['candidate_cell_count'])
    print('trusted_free_cells:', validation['trusted_free_cell_count'])
    print('trusted_free_exported_as_free_cells:', validation['trusted_free_exported_as_free_cell_count'])
    print('uncertainty_cells:', validation['uncertainty_cell_count'])
    print('uncertainty_exported_as_free_cells:', validation['uncertainty_exported_as_free_cell_count'])
    print('conservative_uncertainty_semantics_valid:', validation['conservative_uncertainty_semantics_valid'])
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
