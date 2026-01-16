---
sidebar_position: 9
---

# Model Quantization

This guide explains how to quantize trained Spectrum Analyzer models to different precision levels for deployment optimization.

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)  
**Pre-trained Models:** [Google Drive](https://drive.google.com/drive/folders/1z-cGQhVtElTe6ZCqa5bihQ8guwyl9kaG?usp=sharing)

## Overview

Model quantization reduces the precision of model weights and activations to decrease model size and improve inference speed, making models more suitable for deployment on resource-constrained devices.

The `quantize_model.py` script supports multiple quantization types:
- **FP16**: Half precision floating point
- **Dynamic INT8**: Weights quantized, activations remain FP32
- **Static INT8**: Both weights and activations quantized
- **Per-Channel INT8**: Static INT8 with per-channel quantization scales

## Why Quantize?

**Benefits:**
- **Reduced Model Size**: 2-4x compression
- **Faster Inference**: Especially on CPUs and edge devices
- **Lower Memory Usage**: Important for deployment
- **Better Energy Efficiency**: Reduced computation requirements

**Trade-offs:**
- Slight accuracy degradation (typically &lt;2% mAP)
- Some quantization types require calibration data
- May need hardware-specific optimizations

## Supported Quantization Types

### 1. FP16 (Half Precision)

**Characteristics:**
- ~2x compression
- Minimal accuracy loss
- Fastest quantization (no calibration needed)
- Works on both CPU and GPU
- Best for GPU inference with Tensor Cores

**When to Use:**
- GPU deployment (V100, A100, RTX series)
- When you need fast quantization without calibration
- When accuracy is critical

### 2. Dynamic INT8

**Characteristics:**
- ~4x compression
- Good accuracy
- No calibration needed
- Fast inference
- Weights quantized, activations remain FP32

**When to Use:**
- Quick quantization without calibration data
- CPU inference
- When you need good compression with minimal setup

### 3. Static INT8

**Characteristics:**
- ~4x compression
- Better accuracy than dynamic INT8
- Requires calibration data
- Best for production deployment
- Both weights and activations quantized

**When to Use:**
- Production deployment
- When you have calibration data available
- CPU inference
- Maximum compression with good accuracy

### 4. Per-Channel INT8

**Characteristics:**
- ~4x compression
- Best accuracy among INT8 methods
- Requires calibration data
- Slightly slower than static INT8
- Per-channel quantization scales

**When to Use:**
- When accuracy is critical
- Production deployment
- When you can afford slightly slower inference
- Maximum accuracy with INT8 quantization

## Usage

### Basic Usage

**FP16 Quantization (Simplest):**
```bash
python quantize_model.py \
    --weights runs/train/exp28/weights/best.pt \
    --quant-type fp16 \
    --output quantized_models/
```

**Dynamic INT8 Quantization:**
```bash
python quantize_model.py \
    --weights runs/train/exp28/weights/best.pt \
    --quant-type dynamic_int8 \
    --output quantized_models/
```

### Static Quantization (Requires Calibration)

**Static INT8:**
```bash
python quantize_model.py \
    --weights runs/train/exp28/weights/best.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml \
    --output quantized_models/ \
    --num-calib-batches 50 \
    --batch-size 16 \
    --imgsz 512
```

**Per-Channel INT8 (Best Accuracy):**
```bash
python quantize_model.py \
    --weights runs/train/exp28/weights/best.pt \
    --quant-type per_channel_int8 \
    --data data/raddet.yaml \
    --output quantized_models/ \
    --num-calib-batches 50 \
    --batch-size 16 \
    --imgsz 512
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--weights` | str | Path to trained model weights (.pt file) |
| `--quant-type` | str | Quantization type: `fp16`, `dynamic_int8`, `static_int8`, or `per_channel_int8` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output` | str | `quantized_models` | Output directory for quantized models |
| `--data` | str | None | Dataset YAML file (required for static/per-channel INT8) |
| `--device` | str | `cpu` | Device to use (`cpu` or `cuda`) |
| `--imgsz` | int | 640 | Image size for calibration |
| `--batch-size` | int | 16 | Batch size for calibration data |
| `--num-calib-batches` | int | 100 | Number of batches for calibration (static quantization) |
| `--workers` | int | 4 | Number of dataloader workers |

### Parameter Notes

**Calibration Data:**
- Required for `static_int8` and `per_channel_int8`
- Uses training data from dataset YAML
- More batches = better accuracy but longer calibration time
- Recommended: 50-100 batches

**Image Size:**
- Should match training/inference image size
- Default: 640 (adjust to match your model)

**Device:**
- `cpu`: For CPU inference optimization
- `cuda`: For GPU quantization (FP16 works best)

## Quantization Process

### FP16 Quantization

1. Load model
2. Convert to half precision (FP16)
3. Save quantized model

**No calibration needed** - fastest quantization method.

### Dynamic INT8 Quantization

1. Load model
2. Quantize weights to INT8 (activations remain FP32)
3. Save quantized model

**No calibration needed** - good balance of speed and accuracy.

### Static INT8 Quantization

1. Load model
2. Load calibration data from dataset
3. Prepare model for calibration
4. Run calibration pass (collect statistics)
5. Convert to quantized model
6. Save quantized model

**Requires calibration** - best accuracy for INT8.

### Per-Channel INT8 Quantization

1. Load model
2. Set per-channel quantization config
3. Load calibration data
4. Prepare and calibrate model
5. Convert to quantized model
6. Save quantized model

**Requires calibration** - best accuracy among INT8 methods.

## Output

### Output Files

Quantized models are saved with naming convention:
```
{original_model_name}_{quant_type}.pt
```

**Examples:**
- `best_fp16.pt`
- `best_dynamic_int8.pt`
- `best_static_int8.pt`
- `best_per_channel_int8.pt`

### Output Directory Structure

```
quantized_models/
├── best_fp16.pt
├── best_dynamic_int8.pt
├── best_static_int8.pt
└── best_per_channel_int8.pt
```

### Compression Statistics

The script displays:
- Original model size
- Quantized model size
- Compression ratio
- Quantization type

**Example Output:**
```
Quantization Summary:
  Original model: runs/train/exp28/weights/best.pt
  Quantized model: quantized_models/best_static_int8.pt
  Quantization type: static_int8
  Original size: 45.23 MB
  Quantized size: 11.31 MB
  Compression ratio: 4.00x
```

## Loading Quantized Models

### FP16 Model

```python
import torch
from models.experimental import attempt_load

# Load FP16 model
checkpoint = torch.load('quantized_models/best_fp16.pt', map_location='cpu')
model = checkpoint['model']
model = model.half()  # Ensure FP16
model.eval()

# Inference (input must also be FP16)
imgs = imgs.half()
pred = model(imgs)
```

### INT8 Models

```python
import torch
from models.experimental import attempt_load

# Load INT8 model
checkpoint = torch.load('quantized_models/best_static_int8.pt', map_location='cpu')
model = checkpoint['model']
model.eval()

# Inference (input should be FP32, will be quantized automatically)
imgs = imgs.float() / 255.0
pred = model(imgs)
```

## Best Practices

### Choosing Quantization Type

**For GPU Deployment:**
- Use **FP16** for best performance on Tensor Core GPUs
- Minimal accuracy loss
- Fast inference

**For CPU Deployment:**
- Use **Static INT8** or **Per-Channel INT8**
- Maximum compression
- Good accuracy with calibration

**For Quick Testing:**
- Use **Dynamic INT8**
- No calibration needed
- Good compression

### Calibration Data

**Best Practices:**
- Use representative training data
- 50-100 batches is usually sufficient
- More batches = better accuracy (diminishing returns)
- Ensure calibration data matches inference distribution

**Example:**
```bash
# Use more batches for better accuracy
python quantize_model.py \
    --weights model.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml \
    --num-calib-batches 100 \
    --batch-size 16
```

### Validation

**Always validate quantized models:**
1. Quantize the model
2. Run validation on test set
3. Compare accuracy with original model
4. Accept if accuracy loss is acceptable (&lt;2% mAP typically)

```bash
# Validate quantized model
python val.py \
    --weights quantized_models/best_static_int8.pt \
    --data data/raddet.yaml \
    --task test \
    --imgsz 512
```

## Integration with Model Conversion

Quantization is typically performed after model conversion:

1. **Train Model** (with SiLU activations)
2. **Convert to Binary Spiking** (for neuromorphic hardware)
3. **Quantize Model** (for deployment optimization)

**Example Workflow:**
```bash
# Step 1: Train
python train.py --data data/raddet.yaml --cfg models/resnet18.yaml --epochs 300

# Step 2: Convert to binary spiking
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/model_binary.pt \
    --data data/raddet.yaml

# Step 3: Quantize
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml \
    --output quantized_models/
```

## Performance Considerations

### Model Size Reduction

| Quantization Type | Compression | Typical Size Reduction |
|-------------------|-------------|------------------------|
| FP16 | ~2x | 50% smaller |
| Dynamic INT8 | ~4x | 75% smaller |
| Static INT8 | ~4x | 75% smaller |
| Per-Channel INT8 | ~4x | 75% smaller |

### Inference Speed

- **FP16**: Fastest on GPU with Tensor Cores
- **INT8**: Faster on CPU, optimized for mobile/edge
- **Per-Channel**: Slightly slower than static INT8

### Accuracy Impact

- **FP16**: Minimal loss (&lt;0.5% mAP typically)
- **Dynamic INT8**: Small loss (1-2% mAP)
- **Static INT8**: Small loss (1-2% mAP)
- **Per-Channel INT8**: Smallest loss (&lt;1% mAP)

## Troubleshooting

### Error: "Could not set fbgemm qconfig"

**Problem:** PyTorch wasn't built with FBGEMM support.

**Solutions:**
- Try using `--device cuda` for GPU quantization
- Install PyTorch with FBGEMM support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### Error: "calibration_loader required"

**Problem:** Calibration data is required for static/per-channel INT8.

**Solution:**
- Provide `--data` argument with dataset YAML file
- Ensure dataset path is correct in YAML file

### Error: "Model file not found"

**Problem:** Incorrect path to model weights.

**Solution:**
- Check that the model file exists
- Use absolute path if relative path doesn't work
- Verify file permissions

### Custom Layers (Snn_Conv2d)

**Note:** The script handles standard PyTorch layers automatically. Custom layers may need special quantization handlers. If you encounter issues:
- Try FP16 quantization first (most compatible)
- Check if custom layers are properly registered
- Consider using dynamic INT8 (more compatible)

### Accuracy Degradation

**If accuracy loss is too high:**
- Use more calibration batches
- Try per-channel INT8 instead of static INT8
- Use FP16 if accuracy is critical
- Validate calibration data is representative

## Example: Complete Quantization Workflow

### Step 1: Train Model

```bash
python train.py \
    --data data/raddet.yaml \
    --cfg models/resnet18.yaml \
    --imgsz 512 \
    --epochs 300 \
    --batch-size 64 \
    --device 0
```

### Step 2: Convert to Binary Spiking (Optional)

```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/model_binary.pt \
    --data data/raddet.yaml \
    --img-size 512
```

### Step 3: Quantize Model

```bash
# For GPU deployment (FP16)
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type fp16 \
    --output quantized_models/ \
    --device cuda

# For CPU deployment (Static INT8)
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml \
    --output quantized_models/ \
    --num-calib-batches 50 \
    --batch-size 16 \
    --imgsz 512 \
    --device cpu
```

### Step 4: Validate Quantized Model

```bash
python val.py \
    --weights quantized_models/model_binary_static_int8.pt \
    --data data/raddet.yaml \
    --task test \
    --imgsz 512
```

## Related Documentation

- [Model Conversion](model-conversion.md): Convert SiLU to binary spiking (typically done before quantization)
- [Configuration](configuration.md): Quantization configuration options
- [Training](training.md): Training models before quantization
- [Detection](detection.md): Using quantized models for inference
- [Architecture](architecture.md): Understanding model structure
- [RadDet Use Case](usecase-raddet.md): Example quantization workflow

