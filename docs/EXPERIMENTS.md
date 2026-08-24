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

Status: running

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
corridor extraction
        |
        v
centerline generation
```

Algorithm:

- PCA based dominant direction estimation
- geometric corridor baseline
- centerline export

Current output:

```
results/EXP002/

├── metadata.yaml
├── height.png
├── relative_height.png
├── traversability.png
├── corridor.png
├── centerline.png
└── centerline.csv
```

Future improvements:

- direction consistency
- corridor width constraint
- row parallel structure detection
