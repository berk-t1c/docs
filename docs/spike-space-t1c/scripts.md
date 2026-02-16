---
sidebar_position: 21
---

# Scripts Reference

All executable scripts live in `scripts/` and are run from the repository root.

## train.py -- STDP Training

Layer-wise STDP training with WTA competition and adaptive thresholds.

```bash
python scripts/train.py --config configs/config.yaml [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | -- | YAML config file |
| `--paper` | -- | Paper preset (`igarss2023`, `kheradpisheh2018`) |
| `--output, -o` | `./runs` | Output directory |
| `--name, -n` | auto-generated | Experiment name |
| `--resume, -r` | -- | Resume from checkpoint |
| `--epochs` | 50 | Maximum epochs per layer |
| `--n-classes` | 1 | Number of output classes |
| `--device` | `cuda` | `cuda`, `cpu`, `cuda:0` |
| `--seed` | 42 | Random seed |
| `--log-level` | `INFO` | Logging verbosity |
| `--no-tensorboard` | false | Disable TensorBoard |

**Outputs:** checkpoints (`checkpoint_best.pt`, `checkpoint_latest.pt`, per-epoch), `metrics_final.json`, TensorBoard logs.

---

## train_cv.py -- Cross-Validation

k-fold cross-validation with held-out test set.

```bash
python scripts/train_cv.py --config configs/config.yaml --n-folds 10 [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | `configs/config.yaml` | Config file |
| `--data-root, -d` | from config | EBSSA data root |
| `--n-folds, -k` | 10 | Number of folds |
| `--test-ratio` | 0.1 | Held-out test fraction |
| `--seed` | 42 | Split seed |
| `--output-dir, -o` | `runs/cv` | Output directory |
| `--threshold` | 0.05 | Inference threshold |
| `--eval-only` | false | Skip training, evaluate existing folds |

**Outputs:** `fold_info.json`, `test_recordings.txt`, per-fold directories, `cv_results.json`.

---

## evaluate.py -- Model Evaluation

Pixel-level, object-level, or volume-based evaluation.

```bash
python scripts/evaluate.py --checkpoint best.pt --data-root /path/to/EBSSA [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint, -c` | *required* | Checkpoint path |
| `--data-root, -d` | -- | EBSSA root |
| `--split` | `val` | `train`, `val`, `test`, `all` |
| `--sensor` | `all` | `ATIS`, `DAVIS`, `all` |
| `--volume-based` | false | IGARSS 2023 methodology |
| `--object-level` | false | Centroid matching |
| `--spatial-tolerance` | 1 | TP tolerance in pixels |
| `--no-hulk` | false | Disable HULK |
| `--no-smash` | false | Disable SMASH |
| `--output, -o` | -- | Save metrics JSON |
| `--debug` | false | Debug logging |

---

## inference.py -- Detection and Segmentation

```bash
python scripts/inference.py --checkpoint best.pt --input recording.mat [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint, -c` | *required* | Checkpoint path |
| `--input, -i` | -- | Single `.mat` file |
| `--data-root, -d` | -- | Batch mode: EBSSA root |
| `--output, -o` | `detections.json` | Output JSON |
| `--threshold` | 0.05 | Inference threshold |
| `--visualize` | false | Save 2D images |
| `--visualize-3d` | false | 3D trajectory plots |
| `--device` | auto | `cuda` or `cpu` |

**Outputs:** `detections.json`, optional `detection_*.png` and `figure4_sample_*.png`.

---

## threshold_sweep.py -- Threshold Optimization

Sweeps inference thresholds to find optimal informedness.

```bash
python scripts/threshold_sweep.py --checkpoint best.pt --data-root /path/to/EBSSA
```

Tests thresholds: `[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]`. Prints a table with sensitivity, specificity, and informedness for each, plus the best threshold.

---

## diagnose.py -- Diagnostic Visualization

```bash
python scripts/diagnose.py --checkpoint best.pt --data-root /path/to/EBSSA --n-samples 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | hard-coded path | Checkpoint |
| `--data-root` | hard-coded path | EBSSA root |
| `--n-samples` | 5 | Number of samples |

**Output:** `diagnostic_output.png` -- 5-column grid per sample (input events, GT, raw spikes, overlay, classification result).

---

## demo.py -- Quick Demonstration

```bash
python scripts/demo.py
```

Placeholder script for running a quick forward-pass demonstration. Currently empty; implement to showcase the full pipeline on a sample recording.
