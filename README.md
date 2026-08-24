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

## EXP002 corridor recovery

EXP002 compares three geometry-only stages on the same grid input:

- A: connected traversable free space
- B: component-wise PCA filtering aligned with the recovered crop-row axis
- C: row-frame valley extraction with width, length, and continuity constraints

Run all three stages:

```bash
python tools/run_corridor_test.py \
  --pcd /data/FAST-LIVO2/processed.pcd \
  --mode all \
  --hash-pcd
```

Every invocation creates an immutable directory under
`results/EXP002/<UTC timestamp>_<git commit>/`. See
`docs/EXPERIMENTS.md` for the canonical experiment record and artifact
layout.

## EXP003 ground evidence map

EXP003 replaces a binary traversability assumption with four explicit evidence
states: unknown, confirmed free, confirmed occupied, and bounded interpolated
ground. It also emits a conservative navigation costmap with obstacle
inflation. Ground classification requires independent neighboring support,
and interpolation is limited to small enclosed unsupported components with
local multi-side ground support.

Run the reproducible pipeline:

```bash
python tools/run_ground_evidence_test.py \
  --pcd /data/FAST-LIVO2/processed.pcd \
  --output results/EXP003 \
  --hash-pcd
```

Each invocation creates an immutable run directory containing authoritative
NumPy arrays, YAML provenance and metrics, and PNG inspection aids. The code
and synthetic verification are complete; the authoritative 85M-point run and
throughput profiling remain pending. See `docs/EXPERIMENTS.md` for the exact
configuration, artifact schema, and acceptance boundary.

## Design principle

This repository is independent from navigation runtime. It focuses on evaluating map reconstruction quality before integration into planners.
