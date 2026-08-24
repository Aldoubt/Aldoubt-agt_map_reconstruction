# Development Log

## Project: agt_map_reconstruction

Purpose:

A standalone agricultural LiDAR map reconstruction benchmark. The project evaluates different point cloud processing and ground segmentation methods and converts LiDAR maps into navigation-related 2D representations.

---

## Phase 0 - Benchmark Foundation (Completed)

### Goal

Create a unified benchmark pipeline from PCD input to algorithm comparison output.

### Completed

- PCD loading from FAST-LIVO2/LIO-SLAM generated maps
- Unified segmentation interface
- Baseline algorithm execution framework
- Visualization pipeline

### Current algorithms

| Algorithm | Status | Notes |
|---|---|---|
| Height threshold | completed | Simple global height baseline |
| Morphological PMF baseline | completed | Local morphology inspired baseline |

---

## Phase 1 - First Real Dataset Validation (Completed)

### Dataset

Input:

```
FAST-LIVO2 processed.pcd
size: ~2.6GB
```

### Observation

The first visualization showed that simple point projection was insufficient because all point classes were rendered with the same color.

Changes:

- Added colored segmentation visualization
- Added height grid visualization
- Started evaluating agricultural structure recovery

---

## Phase 2 - Agricultural Structure Recovery (Current)

### Problem discovered

Traditional ground segmentation alone is insufficient for agricultural scenes.

Agricultural environments contain:

- crop rows
- ridges
- corridors
- vegetation above ground
- stable walls

The navigation requirement is not simply:

```
ground / non-ground
```

but:

```
PCD
 |
terrain understanding
 |
traversability reasoning
 |
row corridor extraction
 |
centerline generation
```

---

## Next implementation targets

### 1. Elevation normalization

Convert absolute height into relative height:

```
point_z - local_ground_height
```

Purpose:

Separate crop ridges from actual obstacles.

---

### 2. Traversability grid

Generate:

- free space probability
- obstacle probability
- unknown region

Output:

```
traversability_map.png
traversability.yaml
```

---

### 3. Agricultural corridor extraction

Recover:

- row direction
- aisle regions
- centerline

Output:

```
centerline.csv
corridor.geojson
```

---

## Experiment record rule

Every algorithm update should record:

1. Algorithm name
2. Input dataset
3. Parameters
4. Output visualization
5. Failure cases
6. Decision for next iteration

