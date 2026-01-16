---
sidebar_position: 3
---

# Architecture

This document describes the architecture and general flow of the Spectrum Analyzer system.

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)

## System Overview

The Spectrum Analyzer uses a modified YOLO (You Only Look Once) architecture based on the EMS-YOLO framework. The system processes spectrogram images to detect and classify various signal types.

## General Flow

```
Input Spectrogram Image (512×512)
    ↓
Preprocessing & Normalization
    ↓
Time Window Processing (5 frames)
    ↓
ResNet Backbone (Feature Extraction)
    ↓
Multi-scale Feature Maps
    ↓
YOLO Detection Head
    ↓
Bounding Box Predictions + Class Scores
    ↓
Non-Maximum Suppression (NMS)
    ↓
Output: Detected Signals with Bounding Boxes
```

## Architecture Components

### 1. Input Processing

**Image Input:**
- Default resolution: 512×512 pixels
- Channels: 3 (RGB spectrogram representation)
- Configurable via `--imgsz` parameter

**Time Window Integration:**
- Processes 5-frame temporal windows
- Each frame is duplicated to create temporal context
- Helps capture temporal signal characteristics

### 2. Backbone Network

The backbone extracts features from input images. Supported architectures:

**ResNet-18 (Default):**
- BasicBlock layers for feature extraction
- Multi-scale feature maps at different resolutions
- Configurable via model YAML files

**Other Options:**
- ResNet-10: Lighter model
- ResNet-34: Deeper model
- Custom architectures via YAML configuration

*Model architecture details are available in the training output directory.*

### 3. Detection Head

**YOLO Detection Layers:**
- Two detection scales:
  - P4/16: Medium-scale detection (stride 16)
  - P5/32: Large-scale detection (stride 32)
- Each layer outputs bounding boxes, confidence, and class predictions

**Anchor Boxes:**
- Predefined anchor boxes optimized for signal detection
- P4/16 anchors: [10,14, 23,27, 37,58]
- P5/32 anchors: [81,82, 135,169, 344,319]
- Configurable in model YAML files

### 4. Output Processing

**Detection Output:**
- Bounding boxes (x, y, width, height)
- Confidence scores (0.0-1.0)
- Class predictions (11 classes for RadDet)

**Post-processing:**
- Non-Maximum Suppression (NMS) to remove duplicate detections
- Confidence threshold filtering
- Class-specific filtering (optional)

## Model Configuration

Models are configured via YAML files in the `models/` directory:

```yaml
# Example: models/resnet18.yaml
nc: 11  # number of classes
backbone:
  [[-1, 1, Conv_1, [64, 7, 2]],
   [-1, 1, BasicBlock_2, [64, 3, 2]],
   # ... more layers
  ]
head:
  [[-1, 1, Detect, [nc, anchors]],]
```

## Time Window Processing

The system uses a 5-frame time window for temporal context:

1. Input image is duplicated 5 times
2. Each frame is processed through the backbone
3. Features are aggregated across time dimension
4. Final detection uses temporal information

This helps capture:
- Signal continuity over time
- Temporal patterns in spectrograms
- Better detection of time-varying signals

## Feature Extraction Flow

```
Input Image (1, 3, 512, 512)
    ↓
Time Window Expansion (5, 3, 512, 512)
    ↓
ResNet Backbone
    ├─→ Conv + BasicBlock layers
    ├─→ Multi-scale features
    └─→ Feature maps at different resolutions
    ↓
Detection Layers
    ├─→ P4/16 features → Medium detections
    └─→ P5/32 features → Large detections
    ↓
Temporal Aggregation (sum over time dimension)
    ↓
Final Predictions
```

## Detection Process

1. **Feature Extraction**: Backbone extracts multi-scale features
2. **Detection Prediction**: YOLO head predicts boxes, confidence, classes
3. **Temporal Aggregation**: Features aggregated across time window
4. **NMS**: Non-maximum suppression removes duplicates
5. **Filtering**: Confidence and class filtering
6. **Output**: Final detections with bounding boxes

## Performance Characteristics

**Inference Speed:**
- Depends on image size and model architecture
- ResNet-18: ~XX ms per image (512×512)
- Configurable via batch size and device selection

**Memory Usage:**
- Model size: ~XX MB (ResNet-18)
- GPU memory: Depends on batch size
- Configurable via quantization

## Customization

The architecture can be customized:

1. **Model Backbone**: Change in model YAML file
2. **Image Size**: Configure via `--imgsz` parameter
3. **Number of Classes**: Set in dataset YAML file
4. **Anchors**: Modify in model YAML file
5. **Time Window**: Adjust in `models/yolo.py` (time_window variable)

For detailed configuration options, see [Configuration Guide](configuration.md).

## Related Documentation

- [Quick Start](quickstart.md): Get started with the system
- [Configuration](configuration.md): All configuration options
- [Training](training.md): How to train models
- [Detection](detection.md): Running inference
- [Model Conversion](model-conversion.md): Converting models for deployment
- [RadDet Use Case](usecase-raddet.md): Example use case demonstration

