# Experiment Records

This file is the single entry point for all experiment tracking.

Each experiment records:

- dataset
- algorithm version
- parameters
- output path
- metrics
- failure cases
- conclusion

---

## EXP001 Ground Segmentation Benchmark

Status: completed

Dataset:

```text
FAST-LIVO2 processed.pcd
```

Algorithms:

- height threshold
- morphological PMF

Conclusion:

Binary ground segmentation is only an intermediate representation.
Agricultural navigation requires structure recovery.

Output:

```text
results/EXP001/
```

---

## EXP002 Agricultural Corridor Recovery

Status: completed

Goal:

Recover interpretable agricultural navigation corridors from geometric structure.

Pipeline:

```text
relative elevation
        |
        v
row direction estimation
        |
        v
row-aware corridor extraction
        |
        v
centerline generation
```

Algorithm stages:

### EXP002-A Baseline

- PCA dominant direction estimation
- connected free-space extraction
- centerline export

### EXP002-B Row-aware constraint

- dominant row direction consistency
- geometric filtering of inconsistent regions
- no learned semantic model

Output:

```text
results/EXP002/

├── metadata.yaml
├── height.png
├── relative_height.png
├── traversability.png
├── corridor.png
├── centerline.png
└── centerline.csv
```

Conclusion:

Row-aware corridor recovery provides the geometric prior needed for a navigation-oriented map, but corridor masks alone do not answer whether a robot-sized footprint can traverse every aisle.

Handoff:

- EXP003 owns map-server interface validation and robot-equivalent clearance checks.

---

## EXP003 Navigation Map V2 Interface Validation

Status: completed, superseded by EXP003.1 static obstacle policy correction

Goal:

Convert recovered semantic geometry into a navigation-oriented static-map bundle and quantify aisle connectivity.

Original policy:

```text
aisle prior              -> free
ridge / wall             -> hard occupied
obstacle / step / pillar -> candidate layer
```

The original replay produced 20 / 20 aisles at 0.30 m equivalent clearance, but review of the generated PGM showed that pillar cells inside aisle polygons could be promoted to free. That result is therefore retained only as a pre-correction baseline.

---

## EXP003.1 Static Obstacle Policy Correction

Status: implemented and replay-validated on 2026-08-25

Goal:

Prevent permanent greenhouse structure from being erased by the recovered aisle prior while avoiding a return to overly conservative candidate blocking.

Inputs:

```text
semantic_labels.npy
aisle_rectangles.json
```

Reference dataset:

- grid: 912 x 797
- resolution: 0.05 m/cell
- aisles: 20
- pillar cells: 2181

Corrected semantic priority:

```text
unknown default
      |
      v
aisle prior / semantic aisle -> free
      |
      v
ridge / wall / pillar       -> hard occupied (highest priority)

obstacle_candidate / step_candidate -> advisory candidate_mask only
```

The all-candidate-as-unknown policy was tested and rejected for this dataset because it reduced 0.30 m aisle connectivity to 12 / 20. Pillar-only structural blocking preserves substantially more corridor continuity while fixing the observed safety error.

Static-map grayscale contract:

```text
0   = occupied
205 = unknown
254 = free
```

Nav2 YAML contract:

```text
mode: trinary
occupied_thresh: 0.65
free_thresh: 0.196
```

Implementation:

```text
src/agt_map_reconstruction/maps/navigation_export.py
tools/build_navigation_map.py
tests/test_navigation_export.py
```

Default output:

```text
results/EXP003/navigation-map-v2/

├── navigation_base_map.pgm
├── navigation_base_map.yaml
├── candidate_mask.npy
├── static_obstacle_mask.npy
└── validation.json
```

Reference replay command:

```bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
```

Reference replay metrics after the correction:

| clearance radius | equivalent diameter | aisle connectivity |
| ---: | ---: | ---: |
| 0.20 m | 0.40 m | 20 / 20 |
| 0.25 m | 0.50 m | 20 / 20 |
| 0.30 m | 0.60 m | 17 / 20 |
| 0.35 m | 0.70 m | 14 / 20 |
| 0.40 m | 0.80 m | 13 / 20 |
| 0.50 m | 1.00 m | 12 / 20 |

Additional replay metrics:

- canonical gray values: `[0, 205, 254]`
- map-server YAML validation: pass
- pillar cells: 2181
- pillar cells exported as free: 0
- static obstacle semantic validation: pass
- advisory candidate cells: 5829

Failure cases at 0.30 m radius:

- A01
- A15
- A18

Interpretation:

These failures are more informative than the original 20 / 20 result because the corrected map no longer erases structural pillar geometry. They are not yet proof that the physical robot cannot traverse the aisles; circular clearance is only a conservative proxy for the real polygon footprint and vehicle pose.

Conclusion:

EXP003.1 fixes the static-map semantic bug that allowed pillars to become white/free. The corrected map is suitable as the input to robot-scale validation. Obstacle and step candidates remain outside the permanent hard-obstacle layer so local perception can still resolve uncertain geometry at runtime.

Next stage: EXP004 Robot Footprint & Route Validation

1. validate the actual robot polygon footprint against all aisles;
2. use A05 as the first single-aisle route validation case;
3. inspect A01, A15, and A18 as pillar-sensitive failure cases;
4. validate headland exit and A05 -> A06 transition with vehicle kinematic constraints;
5. only after the offline geometry passes, move to Nav2 runtime tests.
