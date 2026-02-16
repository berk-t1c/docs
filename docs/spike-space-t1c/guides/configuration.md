---
sidebar_position: 5
---

# Configuration Guide

All training, model, and data parameters are specified in YAML files under `configs/`.

## Loading Configuration

```python
from spikeseg.config import load_config, merge_configs

config = load_config("configs/config.yaml")

# Override specific keys
config = load_config("configs/config.yaml", overrides={"max_epochs": 100})
```

## Full Configuration Reference

### Experiment

```yaml
experiment_name: "spikeseg_default"
output_dir: "./runs"
seed: 42
device: "cuda"           # "cuda", "cpu", "cuda:0"
max_epochs: 50
train_conv1: false        # Conv1 = fixed DoG filters
train_conv2: true
train_conv3: true
```

### Model Architecture

```yaml
model:
  n_classes: 1
  conv1_channels: 4
  conv2_channels: 36
  kernel_sizes: [5, 5, 7]
  pool1_kernel: 2
  pool1_stride: 2
  pool2_kernel: 2
  pool2_stride: 2
  thresholds: [0.1, 0.1, 0.1]
  leaks: [0.09, 0.01, 0.0]      # 90%, 10%, 0% of threshold
  use_dog_filters: true
```

### STDP Learning

```yaml
stdp:
  lr_plus: 0.04            # LTP rate (IGARSS 2023)
  lr_minus: 0.03           # LTD rate
  weight_min: 0.0
  weight_max: 1.0
  weight_init_mean: 0.8
  weight_init_std: 0.01
  use_soft_bounds: true     # Multiplicative STDP
```

### Winner-Take-All

```yaml
wta:
  mode: "both"             # "global", "local", "both"
  local_radius: 2
  enable_homeostasis: true
  target_rate: 0.1
  homeostasis_lr: 0.001
  threshold_min: 1.0
  threshold_max: 100.0
```

### Homeostasis

```yaml
homeostasis:
  enabled: true
  theta_rest: 0.1
  theta_plus: 0.02
  tau_theta: 500.0
  theta_max: 10.0
  dead_neuron_recovery: true
  dead_threshold: 0.01
  recovery_boost: 0.1
```

### Data

```yaml
data:
  dataset: "ebssa"
  data_root: "/path/to/EBSSA"
  sensor: "all"             # "ATIS", "DAVIS", "all"
  batch_size: 1
  num_workers: 0
  n_timesteps: 10
  input_height: 128
  input_width: 128
  input_channels: 2         # ON/OFF polarity
  normalize: true
  train_ratio: 0.9
  augmentation:
    enabled: true
    flip_horizontal: true
    flip_polarity: false
```

### Convergence

```yaml
convergence:
  min_wins_per_neuron: 10
  target_ratio: 0.90
  patience: 20
  delta_threshold: 0.0001
  check_interval: 100
```

### Checkpointing

```yaml
checkpoint:
  save_dir: "checkpoints"
  save_interval: 1
  keep_last_n: 3
  save_best: true
  save_on_interrupt: true
```

### Logging

```yaml
logging:
  log_dir: "logs"
  log_level: "INFO"
  log_interval: 100
  tensorboard: true
  wandb: false
  wandb_project: "spikeseg"
```
