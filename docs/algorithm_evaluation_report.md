# Algorithm Evaluation Report

## Experiment 001 - Initial Ground Segmentation Comparison

Date:

2026-08

Dataset:

```
FAST-LIVO2 LIO-only processed.pcd
```

Size:

```
~2.6GB
```

---

# Tested Algorithms

## Height Threshold

Result:

```
ground:      9,838,256
non_ground: 76,074,357
```

Observation:

- Strongly biased toward non-ground classification
- Preserves global structure
- Cannot distinguish agricultural ridges from vegetation/obstacles

Conclusion:

Suitable as a simple baseline only.

---

## Morphological PMF Baseline

Result:

```
ground:      38,445,361
non_ground: 47,467,252
```

Observation:

- Better recovery of continuous terrain
- More suitable for agricultural scenes
- Still cannot directly produce navigation corridors

Conclusion:

Keep as first competitive baseline.

---

# Key Findings

## Finding 1

The scene contains clear agricultural structures:

- parallel crop rows
- aisle structures
- stable boundaries

The point cloud quality is sufficient for geometric reconstruction.

---

## Finding 2

Binary ground segmentation is not the final navigation representation.

Required representation:

```
traversability
+
row structure
+
centerline
```

---

# Current Limitations

- No local ground normalization
- No slope estimation
- No corridor recovery
- No centerline extraction

---

# Next Experiment

Phase 2:

Agricultural structure recovery.

Targets:

1. elevation normalization
2. traversability grid
3. corridor extraction
4. centerline generation
