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

Status: implemented and replay-validated on 2026-08-25

Goal:

Convert recovered semantic geometry into a navigation-oriented static-map bundle while keeping conservative obstacle candidates separate from permanent static obstacles.

Inputs:

```text
semantic_labels.npy
aisle_rectangles.json
```

Reference dataset properties for this validation round:

- grid: 912 x 797
- resolution: 0.05 m/cell
- aisles: 20
- ridges: 20

Map policy:

```text
aisle prior                     -> free
ridge / wall                    -> hard occupied
obstacle / step / pillar        -> candidate layer
outside confirmed aisle/geometry -> unknown
```

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

Reference replay metrics from the current semantic assets:

| clearance radius | equivalent diameter | aisle connectivity |
| ---: | ---: | ---: |
| 0.20 m | 0.40 m | 20 / 20 |
| 0.25 m | 0.50 m | 20 / 20 |
| 0.30 m | 0.60 m | 20 / 20 |
| 0.35 m | 0.70 m | 19 / 20 |
| 0.40 m | 0.80 m | 19 / 20 |
| 0.50 m | 1.00 m | 18 / 20 |

Additional replay metrics:

- canonical gray values: `[0, 205, 254]`
- map-server YAML validation: pass
- candidate cells: 8010
- hard obstacle cells: 76311
- base-map free cells: 353885
- base-map unknown cells: 296668
- base-map occupied cells: 76311

Failure cases:

- A03 (`width_m ~= 0.70`) fails at clearance radius >= 0.35 m.
- A01 (`width_m ~= 0.90`) additionally fails at clearance radius 0.50 m.
- These results are static 2D clearance tests, not proof of Ackermann turning feasibility or localization robustness.

Conclusion:

The navigation-map-v2 policy removes conservative candidate geometry from the permanent static obstacle layer while preserving ridge/wall geometry. On the current reconstructed map, all 20 aisles remain connected through a 0.30 m robot-equivalent safety radius. This is sufficient to move from map-shape inspection to controlled single-aisle navigation tests, but it is not yet sufficient to claim full-site Nav2 readiness.

Next stage:

1. review A03 and the R02-R04 region against the source PCD;
2. verify A16 / R16-R17 spacing anomaly as a true wide aisle or a missing ridge;
3. validate the actual robot polygon footprint instead of only circular-equivalent radii;
4. test one aisle end-to-end, then headland exit, then aisle-to-aisle transition;
5. keep dynamic MID360 obstacles in the local costmap rather than baking them into this static base map.
