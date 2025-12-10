---
sidebar_position: 2
---

# Architecture

## Overview

SpikeYoloV8-Tracker is built on the **BICLab SpikeYOLO (ECCV 2024)** architecture, which uses spiking neural networks (SNNs) for energy-efficient object detection on event camera data.

## BICLab SpikeYOLO (ECCV 2024)

### Neuron Type

- **I-LIF (Integer-valued LIF)**: Spiking neurons that operate on integer values
- **Training**: Integer-valued training with spike-driven inference
- **Architecture**: Simplified YOLOv8 with meta SNN blocks

### Key Components

#### 1. MS_DownSampling
Spiking downsampling layers that reduce spatial dimensions while preserving temporal information.

#### 2. MS_ConvBlock
Spiking convolution blocks that process event-based features through time.

#### 3. SpikeSPPF
Spiking spatial pyramid pooling for multi-scale feature extraction.

#### 4. SpikeDetect
Spiking detection head that outputs bounding boxes, class predictions, and tracking features.

## Project Structure

```
Object_Detection&Tracking/
├── ultralytics/                      # Modified BICLab SpikeYOLO implementation
│   └── nn/
│       └── modules
│           ├── yolo_spikformer.py    # Training Layers (With Tracking)(uses multispike)
│           └── yolo_spikformer_bin.py # Inference Layers (With Tracking)(uses D substep binary spikes)
├── config/                           # Configuration files
│   └── config.yaml                   # Main configuration file
├── src/                              # Core source code
│   ├── __init__.py
│   ├── config_loader.py              # Configuration management
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── logging_utils.py              # Unified logging setup
│   └── etram_spikeyolo_tracking.py   # High-level model architecture
├── scripts/                          # Executable scripts
│   ├── training/                     # Training scripts
│   │   ├── __init__.py
│   │   ├── comprehensive_training.py # Main training script
│   │   └── hyperparameter_search.py  # Hyperparameter search
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── __init__.py
│   │   └── targeted_model_evaluation.py # Targeted model evaluation
│   └── utils/                        # Utility scripts
│       ├── __init__.py
│       └── calculate_class_weights.py # Class weight calculation
├── HDF5/                             # Event data files
├── class annotations/                # Training annotations for classes
├── yolo_loss.py                      # Loss functions used for training
└── requirements.txt                  # Dependencies
```

## Data Flow

### Event Processing Pipeline

1. **Event Loading**: Read events from HDF5 files
2. **Temporal Windowing**: Convert continuous events to discrete time windows
3. **Spike Encoding**: Convert events to spike trains
4. **SNN Processing**: Process spikes through spiking neural network layers
5. **Detection**: Generate bounding boxes and class predictions
6. **Tracking**: Associate detections across time using ByteTracker

### Temporal Processing

The architecture preserves temporal information throughout the network:

- **Input**: `[T, B, C, H, W]` - Temporal sequence of spike frames
- **Processing**: Each temporal step processed independently
- **Output**: `[T, B, H*W, features]` - Temporal predictions preserved
- **Loss**: Computed separately for each temporal step

## Tracking Integration

The model outputs dual predictions:

1. **Detection Features**: Bounding boxes, classes, confidence scores
2. **Tracking Features**: Feature embeddings for object association

ByteTracker uses these features to:
- Associate detections across frames
- Handle occlusions and re-identifications
- Maintain consistent track IDs

## Memory Efficiency

The architecture is designed for low memory usage:

- **Streaming Processing**: Events loaded on-demand
- **Dynamic Batching**: Adapts to available memory
- **Integer Operations**: Reduced precision for efficiency
- **Temporal-Aware Processing**: No unnecessary temporal aggregation

## Research Reference

**BICLab SpikeYOLO (ECCV 2024)**
- **Paper**: "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection"
- **Authors**: Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, Guoqi Li
- **Institution**: BICLab, Institute of Automation, Chinese Academy of Sciences
