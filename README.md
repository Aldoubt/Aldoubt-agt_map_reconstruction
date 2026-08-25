# agt_map_reconstruction

Agricultural LiDAR map reconstruction benchmark.

## Goal

Convert LIO-only PCD maps (FAST-LIVO2 / FAST-LIO2 outputs) into navigation-oriented 2D grid maps while comparing different ground segmentation algorithms.

Current scope:

- offline PCD processing
- ground segmentation benchmark
- elevation map generation
- traversability grid generation
- agricultural corridor recovery
- navigation-oriented static map export and validation
- visualization comparison reports

The repository remains independent from the navigation runtime. It produces and validates map assets before they are integrated into planners or Nav2.

## Pipeline

```text
PCD
 |
 +-- preprocessing
 |
 +-- ground segmentation
 |       |- height threshold
 |       |- PMF
 |       |- CSF
 |       |- Patchwork adapter
 |
 +-- elevation / relative elevation
 |
 +-- traversability grid
 |
 +-- row-aware corridor recovery
 |
 +-- semantic geometry
 |       |- aisle prior
 |       |- ridge / wall / pillar static obstacles
 |       `- obstacle / step candidates
 |
 `-- navigation map v2
         |- navigation_base_map.pgm
         |- navigation_base_map.yaml
         |- candidate_mask.npy
         |- static_obstacle_mask.npy
         `- validation.json
```

## Navigation map v2

`navigation_base_map` uses an explicit semantic priority for navigation safety:

- aisle geometry is used as a free-space prior;
- ridge, wall, and pillar labels are static hard obstacles and override the aisle prior;
- obstacle and step candidates remain an advisory candidate layer instead of being permanently burned into the static map;
- the exported PGM uses only `0 / 205 / 254` for occupied / unknown / free;
- the YAML uses Nav2 trinary thresholds with `free_thresh < occupied_thresh`;
- `validation.json` reports aisle connectivity at multiple robot-equivalent clearance radii and verifies that no static obstacle is exported as free.

Build a bundle from the semantic reconstruction outputs:

```bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
```

Default clearance checks use radii:

```text
0.20 0.25 0.30 0.35 0.40 0.50 m
```

These checks validate static-map corridor connectivity. They do not replace vehicle-footprint, local-costmap, localization, headland-turning, or full Nav2 runtime tests.

## Experiment tracking

`docs/EXPERIMENTS.md` is the single experiment record and points to generated output paths. Generated map assets stay under `results/` and are not treated as source code.

## Design principle

This repository evaluates whether recovered agricultural geometry is interpretable and usable as a navigation-map input. Runtime obstacle perception, localization, planners, controllers, and behavior trees belong outside this repository.
