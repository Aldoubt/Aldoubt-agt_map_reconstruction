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

```
FAST-LIVO2 processed.pcd
```

Algorithms:

- height threshold
- morphological PMF

Conclusion:

Binary ground segmentation is only an intermediate representation.
Agricultural navigation requires structure recovery.

Output:

```
results/EXP001/
```

---

## EXP002 Agricultural Corridor Recovery

Status: implementation complete; synthetic validation passed; real PCD run pending

Goal:

Recover interpretable agricultural navigation corridors from geometric structure.

Pipeline:

```
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

- component-wise PCA row direction estimation
- crop-row direction consistency
- removal of short, compact, or cross-row free-space components
- no learned semantic model

### EXP002-C Geometric corridor constraint

- rotate the map into the recovered crop-row frame
- detect lateral high-row bands and the valley between adjacent rows
- constrain corridor width, longitudinal length, and row-boundary continuity
- rotate accepted corridors back to the source grid

Canonical command:

```bash
python tools/run_corridor_test.py \
  --pcd /data/FAST-LIVO2/processed.pcd \
  --mode all \
  --hash-pcd
```

Each run is immutable. The default run ID is
`<UTC timestamp>_<short git commit>`.

The tree below is produced by the canonical `--mode all` run. A diagnostic
`--mode A`, `B`, or `C` run writes only the selected stage directory and a
single-panel `abc_comparison.png`; it must not be used as the A/B/C acceptance
run.

Output:

```text
results/EXP002/<run_id>/

├── metadata.yaml
├── metrics.yaml
├── height.png
├── relative_height.png
├── traversability.png
├── abc_comparison.png
├── A/
│   ├── corridor.png
│   ├── centerline.png
│   └── centerline.csv
├── B/
│   ├── corridor.png
│   ├── centerline.png
│   └── centerline.csv
└── C/
    ├── corridor.png
    ├── centerline.png
    └── centerline.csv
```

Traceability fields:

- repository and exact Git commit
- dirty-worktree flag
- absolute PCD path and byte size
- optional PCD SHA-256
- point count, UTC time, mode, parameters, row direction, and stage metrics
- grid origin, resolution, and shape; centerline CSV includes both cell and
  PCD-frame metric coordinates at cell centers

Validation evidence:

- synthetic horizontal-row corridor: passed
- component-wise PCA with a cross-row distractor: passed
- width/length/continuity constraints: passed
- rotated row-frame probe at ±15° and 30°: passed
- immutable artifact layout and streaming SHA-256: passed

Real-data status:

- authoritative input: FAST-LIVO2 LIO-only `processed.pcd`
- the 85M-point real-data run has not been executed in this cloud workspace
- do not record real-data corridor quality until the generated run directory is inspected

Current limitations and next decision:

- row-height and width priors still require calibration on the authoritative PCD
- no reference centerline is available, so centerline accuracy remains unevaluated
- navigation utility validation remains outside this repository until the global map is interpretable

Next action: execute the canonical command on the real PCD, retain the whole
`results/EXP002/<run_id>/` directory, and update only this file with observed
metrics, failure cases, and the next algorithm decision.
