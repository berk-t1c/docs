---
sidebar_position: 4
---

# Configuration

This document describes all configuration options available in the Spectrum Analyzer codebase.

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)

## Dataset Configuration

Dataset configuration is specified in YAML files in the `data/` directory.

### Dataset YAML Structure

```yaml
# Dataset root directory
path: /path/to/dataset

# Train, validation, and test paths (relative to path)
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 11

# Class names
names: ['Rect', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'Px', 'ZadoffChu', 'LFM', 'FMCW']
```

### Example: RadDet Configuration

```yaml
# data/raddet.yaml
path: /home/ubuntu/RadDet/RadDet40k512HW009Tv2
train: images/train
val: images/val
test: images/test
nc: 11
names: ['Rect', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'Px', 'ZadoffChu', 'LFM', 'FMCW']
```

## Model Configuration

Model architectures are defined in YAML files in the `models/` directory.

### Model YAML Structure

```yaml
# Number of classes
nc: 11

# Depth and width multipliers
depth_multiple: 1.0
width_multiple: 1.0

# Anchor boxes for each detection layer
anchors:
  - [10,14, 23,27, 37,58]  # P4/16
  - [81,82, 135,169, 344,319]  # P5/32

# Backbone layers
backbone:
  [[-1, 1, Conv_1, [64, 7, 2]],
   [-1, 1, BasicBlock_2, [64, 3, 2]],
   # ... more layers
  ]

# Detection head
head:
  [[-1, 1, Detect, [nc, anchors]],]
```

### Available Model Configurations

- `resnet10.yaml`: Lightweight ResNet-10 backbone
- `resnet18.yaml`: ResNet-18 backbone (default)
- `resnet34.yaml`: Deeper ResNet-34 backbone
- `resnet34-cat.yaml`: ResNet-34 with concatenation
- Custom architectures can be created

## Hyperparameters Configuration

Training hyperparameters are defined in `data/hyps/hyp.scratch.yaml`.

### Hyperparameters Structure

```yaml
# Learning rate
lr0: 0.01  # initial learning rate
lrf: 0.1  # final learning rate (lr0 * lrf)

# Optimizer
momentum: 0.937  # SGD momentum
weight_decay: 0.0005  # weight decay

# Learning rate schedule
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights
box: 0.05  # box loss gain
cls: 0.5  # classification loss gain
obj: 1.0  # objectness loss gain

# Data augmentation
hsv_h: 0.015  # HSV-Hue augmentation
hsv_s: 0.7  # HSV-Saturation augmentation
hsv_v: 0.4  # HSV-Value augmentation
degrees: 0.0  # rotation (+/- deg)
translate: 0.1  # translation (+/- fraction)
scale: 0.5  # scale (+/- gain)
shear: 0.0  # shear (+/- deg)
flipud: 0.0  # flip up-down probability
fliplr: 0.5  # flip left-right probability
mosaic: 1.0  # mosaic augmentation probability
mixup: 0.0  # mixup augmentation probability
```

## Training Configuration

Training is configured via command-line arguments in `train.py`.

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | None | Initial weights path |
| `--cfg` | str | `models/resnet10.yaml` | Model configuration file |
| `--data` | str | `data/coco.yaml` | Dataset configuration file |
| `--hyp` | str | `data/hyps/hyp.scratch.yaml` | Hyperparameters file |
| `--epochs` | int | 300 | Number of training epochs |
| `--batch-size` | int | 16 | Batch size |
| `--imgsz` | int | 640 | Image size (pixels) |
| `--rect` | flag | False | Rectangular training |
| `--resume` | str/bool | False | Resume training from checkpoint |
| `--device` | str | `4,5,6,7` | CUDA device(s) or 'cpu' |
| `--workers` | int | 4 | DataLoader workers |
| `--project` | str | `runs/train` | Project directory |
| `--name` | str | `exp` | Experiment name |
| `--cache` | str | None | Cache images in 'ram' or 'disk' |
| `--multi-scale` | flag | False | Vary image size +/- 50% |
| `--single-cls` | flag | False | Train as single-class |
| `--adam` | flag | False | Use Adam optimizer |
| `--freeze` | int | 0 | Number of layers to freeze |
| `--save-period` | int | -1 | Save checkpoint every N epochs |

### Example Training Command

```bash
python train.py \
    --data data/raddet.yaml \
    --cfg models/resnet18.yaml \
    --imgsz 512 \
    --epochs 300 \
    --batch-size 64 \
    --workers 20 \
    --device 0 \
    --cache ram
```

## Validation Configuration

Validation is configured via command-line arguments in `val.py`.

### Validation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | Required | Model weights path |
| `--data` | str | `data/coco.yaml` | Dataset configuration |
| `--batch-size` | int | 32 | Batch size |
| `--imgsz` | int | 640 | Image size (pixels) |
| `--conf-thres` | float | 0.001 | Confidence threshold |
| `--iou-thres` | float | 0.6 | NMS IoU threshold |
| `--task` | str | `val` | Task: train/val/test |
| `--device` | str | `''` | CUDA device or 'cpu' |
| `--augment` | flag | False | Augmented inference |
| `--verbose` | flag | False | Report mAP by class |
| `--save-txt` | flag | False | Save results to *.txt |
| `--save-json` | flag | False | Save COCO-JSON results |

### Example Validation Command

```bash
python val.py \
    --weights runs/train/exp28/weights/best.pt \
    --data data/raddet.yaml \
    --task test \
    --imgsz 512 \
    --conf-thres 0.001
```

## Detection/Inference Configuration

Detection is configured via command-line arguments in `detect.py`.

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | Required | Model weights path |
| `--source` | str | `data/images` | Source: file/dir/URL |
| `--imgsz` | int | 640 | Inference image size |
| `--conf-thres` | float | 0.25 | Confidence threshold |
| `--iou-thres` | float | 0.45 | NMS IoU threshold |
| `--max-det` | int | 1000 | Maximum detections per image |
| `--device` | str | `''` | CUDA device or 'cpu' |
| `--save-txt` | flag | False | Save results to *.txt |
| `--save-img` | flag | False | Save inference images |
| `--save-conf` | flag | False | Save confidences in labels |
| `--classes` | list | None | Filter by class IDs |
| `--agnostic-nms` | flag | False | Class-agnostic NMS |
| `--augment` | flag | False | Augmented inference |
| `--half` | flag | False | FP16 half-precision |
| `--line-thickness` | int | 3 | Bounding box thickness |
| `--hide-labels` | flag | False | Hide labels |
| `--hide-conf` | flag | False | Hide confidences |

### Example Detection Command

```bash
python detect.py \
    --weights runs/train/exp28/weights/best.pt \
    --source path/to/images \
    --imgsz 512 \
    --conf-thres 0.25
```

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | Required | Model weights path |
| `--source` | str | `data/images` | Source: file/dir/URL |
| `--imgsz` | int | 640 | Inference image size |
| `--conf-thres` | float | 0.25 | Confidence threshold |
| `--iou-thres` | float | 0.45 | NMS IoU threshold |
| `--max-det` | int | 1000 | Maximum detections per image |
| `--device` | str | `''` | CUDA device or 'cpu' |
| `--save-txt` | flag | False | Save results to *.txt |
| `--save-img` | flag | False | Save inference images |
| `--save-conf` | flag | False | Save confidences in labels |
| `--classes` | list | None | Filter by class IDs |
| `--agnostic-nms` | flag | False | Class-agnostic NMS |
| `--augment` | flag | False | Augmented inference |
| `--half` | flag | False | FP16 half-precision |
| `--line-thickness` | int | 3 | Bounding box thickness |
| `--hide-labels` | flag | False | Hide labels |
| `--hide-conf` | flag | False | Hide confidences |

### Example Detection Command

```bash
python detect.py \
    --weights runs/train/exp28/weights/best.pt \
    --source path/to/images \
    --imgsz 512 \
    --conf-thres 0.25 \
    --save-img true
```

## Model Conversion Configuration

Model conversion from SiLU to binary spiking is configured in `convert_to_binary_spiking.py`.

### Conversion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | **Required** | Path to trained model weights (.pt file) |
| `--output` | str | `*_binary.pt` | Output path for converted model (adds `_binary` suffix if not specified) |
| `--diagram` | str | `model_architecture.txt` | Output file for architecture diagram |
| `--device` | str | `cuda` | Device to load model on (`cpu` or `cuda`) |
| `--data` | str | None | Path to dataset YAML file for forward pass tracing (recommended for accurate diagram) |
| `--img-size` | int | 640 | Input image size for forward pass tracing (used as fallback if `--data` not provided) |
| `--no-trace` | flag | False | Disable forward pass tracing (show all modules, including unused ones) |

### Conversion Behavior

**Default (without `--data`):**
- Uses dummy data for forward pass
- Shows all layers including unused shortcuts and conditional layers
- May not accurately represent actual execution paths

**With `--data` (Recommended):**
- Uses real test data from dataset
- Performs actual forward pass
- Only shows layers and operations actually executed
- Provides accurate architecture representation

### Example Conversion Commands

**Basic conversion:**
```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt
```

**With forward pass tracing (accurate diagram):**
```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/model_binary.pt \
    --diagram arch/architecture_diagram.txt \
    --data data/raddet.yaml \
    --img-size 512 \
    --device cuda
```

**Show all layers (disable tracing):**
```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --no-trace
```

## Model Quantization Configuration

Quantization is configured in `quantize_model.py`. For detailed quantization guide, see [Quantization Documentation](quantization.md).

### Quantization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weights` | str | **Required** | Path to model weights (.pt file) |
| `--quant-type` | str | **Required** | Quantization type: `fp16`, `dynamic_int8`, `static_int8`, or `per_channel_int8` |
| `--output` | str | `quantized_models` | Output directory for quantized models |
| `--data` | str | None | Dataset YAML file (required for static/per-channel INT8) |
| `--device` | str | `cpu` | Device to use (`cpu` or `cuda`) |
| `--imgsz` | int | 640 | Image size for calibration |
| `--batch-size` | int | 16 | Batch size for calibration data |
| `--num-calib-batches` | int | 100 | Number of batches for calibration (static quantization) |
| `--workers` | int | 4 | Number of dataloader workers |

### Quantization Types

**FP16 (Half Precision):**
- ~2x compression
- No calibration needed
- Best for GPU deployment

**Dynamic INT8:**
- ~4x compression
- No calibration needed
- Weights quantized, activations FP32

**Static INT8:**
- ~4x compression
- Requires calibration data
- Best for production deployment

**Per-Channel INT8:**
- ~4x compression
- Requires calibration data
- Best accuracy among INT8 methods

### Example Quantization Commands

**FP16 Quantization:**
```bash
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type fp16 \
    --output quantized_models/ \
    --device cuda
```

**Static INT8 Quantization:**
```bash
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml \
    --num-calib-batches 50 \
    --batch-size 16 \
    --imgsz 512 \
    --output quantized_models/
```

**Per-Channel INT8 Quantization:**
```bash
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type per_channel_int8 \
    --data data/raddet.yaml \
    --num-calib-batches 50 \
    --batch-size 16 \
    --imgsz 512 \
    --output quantized_models/
```

## Image Size Configuration

Image size is configurable and affects:

- **Model Input**: All images are resized to this size
- **Detection Resolution**: Higher resolution = better detection but slower
- **Memory Usage**: Larger images require more GPU memory

**Common Sizes:**
- 320×320: Fast, lower accuracy
- 512×512: Balanced (default for RadDet)
- 640×640: Higher accuracy, slower
- Custom: Any size divisible by 32 (YOLO requirement)

Set via `--imgsz` parameter in training, validation, and detection.

## Device Configuration

**Single GPU:**
```bash
--device 0
```

**Multiple GPUs:**
```bash
--device 0,1,2,3
```

**CPU:**
```bash
--device cpu
```

## Output Configuration

**Training Output:**
- `runs/train/{name}/weights/`: Model checkpoints
- `runs/train/{name}/results.png`: Training curves
- `runs/train/{name}/PR_curve.png`: Precision-Recall curve
- `runs/train/{name}/P_curve.png`: Precision-Confidence curve
- `runs/train/{name}/R_curve.png`: Recall-Confidence curve

**Detection Output:**
- `runs/detect/{name}/`: Detection result images
- `runs/detect/{name}/labels/`: Detection labels (if `--save-txt`)

**Model Conversion Output:**
- `*_binary.pt`: Converted model with binary spiking activations
- `*_binary_state_dict.pt`: Model state dictionary
- `model_architecture.txt`: Detailed architecture diagram

Configure via `--project` and `--name` parameters.

## Related Documentation

- [Architecture](architecture.md): Understanding the model architecture
- [Training](training.md): Training configuration and usage
- [Detection](detection.md): Detection/inference configuration
- [Model Conversion](model-conversion.md): Converting models for deployment
- [Quantization](quantization.md): Model quantization options
- [RadDet Use Case](usecase-raddet.md): Example configuration for RadDet dataset

