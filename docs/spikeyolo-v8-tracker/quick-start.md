---
sidebar_position: 8
---

# Quick Start Guide

## Prerequisites

Before starting, ensure you have:

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Event camera data in eTraM format
- Annotations in the expected format

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/type1compute/SpikeYoloV8-Tracker.git
cd SpikeYoloV8-Tracker
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
nvidia-smi  # Check GPU availability
```

## Setup

1. **Organize your data**:
```
SpikeYoloV8-Tracker/
├── HDF5/
│   ├── train_h5_6/
│   ├── val_h5_1/
│   └── test_h5_1/
└── class annotations/
    ├── eight_class_annotations_train/
    ├── eight_class_annotations_val/
    └── eight_class_annotations_test/
```

2. **Configure your settings**:
Edit `config/config.yaml` with your paths and parameters.

## Basic Usage

### Training

Run training with default parameters:

```bash
python scripts/training/comprehensive_training.py \
    --slice_duration_us 100 \
    --time_steps 8 \
    --batch_size 8
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluation/targeted_model_evaluation.py \
    --checkpoint_path /path/to/checkpoint.pt
```

### Hyperparameter Search

Run hyperparameter optimization:

```bash
python scripts/training/hyperparameter_search.py
```

## Common Workflows

### Workflow 1: Quick Training Test

Test the training pipeline with a small subset:

```bash
# Use smaller batch size and fewer epochs
python scripts/training/comprehensive_training.py \
    --slice_duration_us 100 \
    --time_steps 8 \
    --batch_size 4 \
    --max_epochs 5
```

### Workflow 2: Full Training Run

Complete training with all optimizations:

```bash
python scripts/training/comprehensive_training.py \
    --slice_duration_us 100 \
    --time_steps 8 \
    --batch_size 25 \
    --max_epochs 25 \
    --learning_rate 0.001 \
    --device cuda
```

### Workflow 3: Evaluation and Analysis

Evaluate and analyze results:

```bash
# Evaluate model
python scripts/evaluation/targeted_model_evaluation.py \
    --checkpoint_path ./checkpoints/best_model.pt

# Check logs
tail -f logs/evaluation_best_model.log
```

## Import Structure

The project uses absolute imports from the `src` package:

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src
from src.config_loader import ConfigLoader
from src.data_loader import create_ultra_low_memory_dataloader
from src.logging_utils import setup_logging
```

## Running Scripts

All scripts should be run from the project root directory:

```bash
# From project root
cd SpikeYoloV8-Tracker
python scripts/training/comprehensive_training.py [args]
```

The scripts automatically handle path resolution.

## Monitoring

### GPU Usage

```bash
# Check GPU status
nvidia-smi

# Real-time monitoring
watch -n 1 nvidia-smi
```

### Training Progress

Logs are automatically saved to `{model.logs_dir}/`:

```bash
# View training log
tail -f logs/training_20240101_120000.log
```

## Next Steps

- [Installation Guide](./installation) - Detailed installation instructions
- [Configuration Guide](./configuration) - Configure your training
- [Training Guide](./training) - Advanced training techniques
- [Dataset Format](./dataset-format) - Understand data requirements

## Troubleshooting

### Issue: Import Errors

**Solution**: Ensure you're running from project root and paths are correct.

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or `max_events_per_sample` in config.

### Issue: No Annotations Found

**Solution**: Check annotation paths and temporal matching buffer.

### Issue: Slow Training

**Solution**: 
- Check GPU utilization
- Reduce `prefetch_factor`
- Use `pin_memory: true` in config

## Getting Help

- **Repository**: [GitHub Issues](https://github.com/type1compute/SpikeYoloV8-Tracker/issues)
- **Documentation**: See other sections in this documentation
- **Logs**: Check log files for detailed error messages
