# EXP_002 Agricultural Structure Recovery

## Dataset

- Input: FAST-LIVO2 LIO-only processed.pcd
- Size: approximately 2.6 GB
- Scene: greenhouse agricultural environment

---

## Motivation

The first benchmark showed that binary ground segmentation is not enough for agricultural navigation.

The required representation is:

```
PCD
 |
terrain understanding
 |
traversability reasoning
 |
row corridor extraction
```

---

## Previous baseline results

### Height threshold

```
ground: 9838256
non-ground: 76074357
```

Conclusion:

- Strongly biased toward non-ground classification.
- Kept as a simple baseline.

### Morphological PMF baseline

```
ground: 38445361
non-ground: 47467252
```

Conclusion:

- Better terrain continuity.
- Row-like structures become visible.

---

## Current analysis

Observed structures:

- parallel agricultural rows
- greenhouse boundary structures
- possible corridors between rows

Current limitation:

- Height maps describe terrain but do not directly describe robot accessibility.

---

## Next experiment plan

### Step 1: Elevation normalization

Generate relative height:

```
relative_height = point_z - local_ground_height
```

### Step 2: Traversability reasoning

Estimate:

- free probability
- obstacle probability
- roughness
- unknown area

### Step 3: Corridor extraction

Recover:

- row orientation
- corridor mask
- centerline

---

## Decision

Do not continue adding segmentation algorithms before completing agricultural structure reasoning. The benchmark should evaluate the complete map reconstruction pipeline rather than only ground/non-ground accuracy.
