# Agricultural LiDAR Map Reconstruction Benchmark

## Goal

Compare LiDAR ground segmentation and traversability map generation algorithms for agricultural environments.

Input:

- FAST-LIVO2 / LIO-SLAM PCD map

Output:

- ground segmentation
- elevation map
- traversability grid
- visualization report

## Pipeline

PCD
→ preprocessing
→ ground segmentation
→ elevation analysis
→ traversability estimation
→ 2D grid export
→ comparison report

## Algorithm interface

All algorithms expose:

```python
segment(points, config) -> SegmentationResult
```

Result:

- ground_points
- non_ground_points
- metadata

## Phase 1 algorithms

- height threshold baseline
- PMF
- CSF
- Patchwork adapter
- LineFit adapter

## Evaluation

Metrics:

- ground recovery
- wall preservation
- corridor continuity
- traversable area ratio
- centerline quality
