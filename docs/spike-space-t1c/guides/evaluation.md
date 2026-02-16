---
sidebar_position: 3
---

# Evaluation Guide

SpikeSEG supports three evaluation modes, each computing different metrics.

## Quick Start

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --data-root /path/to/EBSSA \
  --volume-based \
  --output metrics.json
```

## Evaluation Modes

### 1. Pixel-Level

Binary classification of each pixel with 1-pixel spatial tolerance.

**Metrics:** TP, TN, FP, FN, Sensitivity, Specificity, Informedness, Accuracy, IoU.

### 2. Object-Level

Centroid matching: a detection is a true positive if its centroid falls within 1 pixel of the ground-truth trajectory.

**Metrics:** TP, FP, FN, Sensitivity (Recall), Specificity, Precision, F1, Informedness.

### 3. Volume-Based (IGARSS 2023)

Event density comparison between predicted and ground-truth regions. This is the **primary evaluation methodology** from the IGARSS 2023 paper [4].

**Metrics:** TP/TN/FP/FN volumes, Sensitivity, Specificity, Informedness, Precision, F1.

**Target:** 89.1% informedness [4].

$$
\text{Informedness} = \text{Sensitivity} + \text{Specificity} - 1
$$

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint, -c` | *required* | Model checkpoint |
| `--data-root, -d` | -- | EBSSA dataset root |
| `--split` | `val` | `train`, `val`, `test`, `all` |
| `--sensor` | `all` | `ATIS`, `DAVIS`, `all` |
| `--volume-based` | false | Use IGARSS 2023 methodology |
| `--object-level` | false | Use centroid matching |
| `--spatial-tolerance` | 1 | Tolerance in pixels for TP matching |
| `--no-hulk` | false | Disable HULK decoder |
| `--no-smash` | false | Disable SMASH grouping |
| `--output, -o` | -- | Save metrics to JSON |
| `--debug` | false | Enable debug logging |

## Threshold Sweep

Find the optimal inference threshold:

```bash
python scripts/threshold_sweep.py \
  --checkpoint checkpoints/best.pt \
  --data-root /path/to/EBSSA
```

This sweeps thresholds `[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]` and reports sensitivity, specificity, and informedness for each, highlighting the best.
