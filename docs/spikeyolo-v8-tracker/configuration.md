---
sidebar_position: 5
---

# Configuration Guide

## Overview

SpikeYoloV8-Tracker uses a centralized configuration system through `config/config.yaml`. All project settings are managed in one place, with no hardcoded values.

## Configuration File Structure

The main configuration file is located at `config/config.yaml`. It contains sections for:

- **Model Configuration**: Architecture and model parameters
- **Training Configuration**: Training hyperparameters and settings
- **Data Processing**: Data loading and preprocessing settings
- **Loss Configuration**: Loss function weights and parameters
- **Logging Configuration**: Logging settings and paths

## Key Configuration Sections

### Model Configuration

```yaml
model:
  name: "SpikeYOLO_Tracker"
  num_classes: 8  # Automatically detected from classes list
  classes:
    - Pedestrian
    - Car
    - Bicycle
    - Bus
    - Motorbike
    - Truck
    - Tram
    - Wheelchair
  logs_dir: "./logs"
```

### Training Configuration

```yaml
training:
  epochs: 25
  warmup_epochs: 3
  batch_size: 25
  learning_rate: 0.001
  optimizer: "sgd"  # Options: "sgd", "adamw"
  momentum: 0.9
  lr_scheduler: "step"  # Options: "step", "cyclic"
  max_learning_rate: 0.002  # For cyclic scheduler
  device: "cuda"  # Options: "cuda", "cpu"
```

### Data Processing Configuration

```yaml
data_processing:
  max_samples_per_file: 50
  max_events_per_sample: 10000
  temporal_buffer: 0.2  # 20% buffer for temporal matching
  slice_duration_us: 100  # Time window in microseconds
  time_steps: 8  # Number of temporal steps
  prefetch_factor: 2
  persistent_workers: true
  pin_memory: true
```

### Loss Configuration

```yaml
yolo_loss:
  box_loss_weight: 5.0  # Increased from 2.0 for better localization
  class_loss_weight: 1.0
  obj_loss_weight: 1.0
  track_loss_weight: 1.0
  use_focal_loss: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  label_smoothing: 0.1  # Prevents overconfidence
```

## Configuration Access

The project uses `src/config_loader.py` for configuration management:

```python
from src.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader('config/config.yaml')

# Access configuration values
num_classes = config.get('model.num_classes')
batch_size = config.get('training.batch_size')
learning_rate = config.get('training.learning_rate')
```

## Dynamic Class Configuration

Classes are defined dynamically in the configuration file. The number of classes is automatically detected:

```yaml
model:
  classes:
    - Pedestrian
    - Car
    - Bicycle
    # Add more classes as needed
```

The system automatically:
- Detects the number of classes from the list
- Updates model architecture accordingly
- Handles class mappings in annotations

## Training Optimizations Configuration

### Warmup Learning Rate

```yaml
training:
  warmup_epochs: 3  # Gradual LR increase prevents cold start
```

### Cyclic Learning Rate

```yaml
training:
  lr_scheduler: "cyclic"
  max_learning_rate: 0.002
  # OneCycleLR scheduler helps escape poor local minima
```

### Focal Loss

```yaml
yolo_loss:
  use_focal_loss: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  # Focuses learning on hard-to-classify examples
```

### Label Smoothing

```yaml
yolo_loss:
  label_smoothing: 0.1
  # Softens hard labels to prevent overconfidence
```

## Data Loader Configuration

```yaml
data_processing:
  prefetch_factor: 2  # Number of batches to prefetch
  persistent_workers: true  # Keep workers alive between epochs
  pin_memory: true  # Faster GPU transfer
```

## Logging Configuration

```yaml
model:
  logs_dir: "./logs"
  # Logs are automatically created in this directory:
  # - training_{timestamp}.log
  # - evaluation_{checkpoint_name}.log
  # - hyperparameter_search/hyperparameter_search_{trial_name}.log
```

## Configuration Best Practices

1. **Centralized Management**: All settings in `config.yaml`
2. **No Hardcoded Values**: Use config loader for all parameters
3. **Dynamic Classes**: Define classes in config, not in code
4. **Version Control**: Keep config files in version control
5. **Environment-Specific**: Use different configs for dev/prod if needed

## Example: Complete Configuration

```yaml
model:
  name: "SpikeYOLO_Tracker"
  classes:
    - Pedestrian
    - Car
    - Bicycle
    - Bus
    - Motorbike
    - Truck
    - Tram
    - Wheelchair
  logs_dir: "./logs"

training:
  epochs: 25
  warmup_epochs: 3
  batch_size: 25
  learning_rate: 0.001
  optimizer: "sgd"
  momentum: 0.9
  lr_scheduler: "step"
  device: "cuda"

data_processing:
  max_samples_per_file: 50
  max_events_per_sample: 10000
  temporal_buffer: 0.2
  slice_duration_us: 100
  time_steps: 8
  prefetch_factor: 2
  persistent_workers: true
  pin_memory: true

yolo_loss:
  box_loss_weight: 5.0
  class_loss_weight: 1.0
  obj_loss_weight: 1.0
  track_loss_weight: 1.0
  use_focal_loss: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  label_smoothing: 0.1
```

## Next Steps

- [Training Guide](./training)
- [Quick Start Guide](./quick-start)
