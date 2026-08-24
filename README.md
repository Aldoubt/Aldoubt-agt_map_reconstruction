# agt_map_reconstruction

Agricultural LiDAR map reconstruction benchmark.

## Goal

Convert LIO-only PCD maps (FAST-LIVO2 / FAST-LIO2 outputs) into navigation-oriented 2D grid maps while comparing different ground segmentation algorithms.

Current scope:

- offline PCD processing
- ground segmentation benchmark
- elevation map generation
- traversability grid generation
- visualization comparison reports

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
 +-- elevation map
 |
 +-- traversability grid
 |
 +-- visualization report
```

## Phase 1 Output

Each algorithm generates:

- ground cloud
- non-ground cloud
- height map
- occupancy/traversability grid
- comparison image

## Design principle

This repository is independent from navigation runtime. It focuses on evaluating map reconstruction quality before integration into planners.
