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

Handoff:

- EXP004 owns polygon-footprint and route-level validation.

---

## EXP004 Robot Footprint & Route Validation

### EXP004-A Polygon Footprint Centerline Baseline

Status: implementation complete; physical-robot acceptance pending measured footprint input

Goal:

Replace circular-equivalent clearance with an explicit polygon footprint and determine whether the recovered aisle centerline itself is collision-free for that footprint.

Inputs:

```text
results/EXP003/navigation-map-v2/navigation_base_map.pgm
results/EXP003/navigation-map-v2/navigation_base_map.yaml
results/EXP003/navigation-map-v2/candidate_mask.npy
aisle_rectangles.json
robot_footprint.json
```

Footprint contract:

```json
{
  "name": "robot_name",
  "polygon_xy_m": [
    [0.50, 0.30],
    [0.50, -0.30],
    [-0.50, -0.30],
    [-0.50, 0.30]
  ]
}
```

Coordinates are metres in the robot `base_link` frame with `+x` forward and `+y` left. The numerical polygon above is an example only and must be replaced with measured geometry before acceptance.

Validation policy:

```text
occupied            -> fail sampled pose
unknown (default)    -> fail sampled pose
candidate_mask       -> report overlap only
out of map           -> fail sampled pose
```

`--allow-unknown` exists only for diagnostic comparison. EXP004-A samples the footprint strictly along the recovered aisle centerline; it does not search for a laterally shifted collision-free route.

Implementation:

```text
src/agt_map_reconstruction/maps/footprint_validation.py
tools/validate_robot_footprint.py
tests/test_footprint_validation.py
```

Output:

```text
results/EXP004/robot-footprint-v1/
├── aisle_footprint_validation.json
└── aisle_footprint_validation.csv
```

Per-aisle metrics:

- sampled pose count
- occupied collision pose count
- unknown overlap pose count
- advisory candidate overlap pose count
- out-of-bounds pose count
- minimum clearance to the active blocking policy
- first failure reason and first failure pose

Test status:

```text
pytest -q tests/test_footprint_validation.py
6 passed
```

Smoke replay on the current corrected 912 x 797 map:

- sample spacing: 0.10 m
- benchmark-only footprint: 1.00 m x 0.60 m centred rectangle
- default unknown-blocking policy: 1 / 20 centerlines pass (`A14`)
- diagnostic `--allow-unknown`: 4 / 20 centerlines pass (`A03`, `A08`, `A12`, `A14`)

This benchmark is **not a physical robot result**. It validates the EXP004-A pipeline and reveals that strict geometric centerlines frequently intersect static structure or unknown cells. EXP003 clearance connectivity can still be higher because it only asks whether some connected free corridor exists and may permit lateral deviation around obstacles.

Conclusion:

EXP004-A is suitable as a strict centerline baseline and as a regression test for any measured polygon footprint. A centerline failure must not be interpreted as aisle-unreachable until route search is allowed to shift laterally inside the aisle.

Next stage: EXP004-B In-Aisle Route Search

1. ingest the measured robot polygon and freeze it as the EXP004 acceptance footprint;
2. search multiple lateral offsets / a collision-free path inside each aisle rather than forcing the recovered centerline;
3. use A05 as the first route case and compare centerline vs recovered collision-free route;
4. only after in-aisle route validation, add headland exit and A05 -> A06 Ackermann transition constraints.
