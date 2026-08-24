# Development Log

## Project: agt_map_reconstruction

Purpose:

A standalone agricultural LiDAR map reconstruction benchmark. The project evaluates different point cloud processing and ground segmentation methods and converts LiDAR maps into navigation-related 2D representations.

---

# Phase 0 - Benchmark Foundation (Completed)

Completed:

- PCD loading from FAST-LIVO2/LIO-SLAM maps
- Unified segmentation interface
- Algorithm comparison framework
- Visualization pipeline

Algorithms:

| Algorithm | Status | Notes |
|---|---|---|
| Height threshold | completed | Global height baseline |
| Morphological PMF baseline | completed | Local morphology baseline |

---

# Phase 1 - Real FAST-LIVO2 Dataset Validation (Completed)

Dataset:

```
FAST-LIVO2 processed.pcd
size: ~2.6GB
```

Results:

## Height threshold

```
ground:      9838256
non-ground: 76074357
```

Observation:

- Excessive non-ground classification.
- Useful only as a simple baseline.

## Morphological PMF baseline

```
ground:      38445361
non-ground: 47467252
```

Observation:

- Better recovery of continuous terrain.
- Agricultural row structures become visible.

---

# Phase 2 - Agricultural Structure Recovery (Completed)

## Motivation

Traditional ground segmentation does not directly solve agricultural navigation.

The target changes from:

```
ground / non-ground
```

into:

```
PCD
 |
terrain understanding
 |
relative elevation
 |
traversability reasoning
 |
row corridor extraction
 |
centerline generation
```

---

# Phase 2.1 - Relative Elevation and Traversability (Implemented)

Added:

- Local elevation normalization
- Relative height calculation
- Initial geometry based traversability map

Pipeline:

```
Ground cloud
    |
height grid
    |
local ground estimation
    |
relative height
    |
traversability classification
```

Outputs:

```
height_map.png
relative_height.png
traversability.png
```

Current classification:

```
0 unknown
1 traversable
2 obstacle
```

Note:

The current traversability model is a baseline and will be improved with:

- slope
- roughness
- corridor continuity
- row structure constraints

---

# Phase 2.2 Corridor Extraction

EXP002 A/B/C corridor recovery is implemented, synthetic-tested, and run on
the authoritative FAST-LIVO2 PCD. A oversegments, B collapses, and C recovers
four plausible local corridors with insufficient global recall. Exact run
identity, input hash, metrics, commands, and conclusions are maintained only
in `docs/EXPERIMENTS.md`.

---

# Phase 3 - Conservative Ground Evidence (Current)

EXP003 implements chunked robust elevation statistics, bounded local ground
interpolation, independently supported ground classification, four-state
evidence, and an inflated navigation costmap. Open, edge-touching, oversized,
or sparsely supported gaps remain unknown. Code, synthetic tests, artifact
checks, and the synthetic CLI smoke are complete.

Pending:

- run the authoritative 85M-point PCD
- inspect real-map evidence and costmap quality
- profile runtime, throughput, and peak memory at acceptance scale

EXP004 rosbag raycasting is deferred as a feasibility probe only. It is not an
implemented capability or current baseline. The canonical command, output
tree, configuration, acceptance boundary, and future decision are maintained
only in `docs/EXPERIMENTS.md`.

---

# Experiment Record Rule

Every algorithm update records:

1. Algorithm name
2. Dataset
3. Parameters
4. Visualization output
5. Failure cases
6. Decision for next iteration
