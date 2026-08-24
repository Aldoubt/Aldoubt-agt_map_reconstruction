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

# Phase 2 - Agricultural Structure Recovery (Current)

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

# Next targets

## Phase 2.2 Corridor Extraction

Implement:

- row direction estimation
- parallel structure detection
- corridor mask generation
- centerline extraction

Outputs:

```
corridor_mask.png
centerline.csv
```

---

# Experiment Record Rule

Every algorithm update records:

1. Algorithm name
2. Dataset
3. Parameters
4. Visualization output
5. Failure cases
6. Decision for next iteration
