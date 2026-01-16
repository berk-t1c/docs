---
sidebar_position: 8
---

# Model Conversion: SiLU to Binary Spiking

This guide covers converting trained models from SiLU activation to binary spiking activation for deployment on neuromorphic hardware.

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)  
**Pre-trained Models:** [Google Drive](https://drive.google.com/drive/folders/1z-cGQhVtElTe6ZCqa5bihQ8guwyl9kaG?usp=sharing)

## Overview

The Spectrum Analyzer model is trained using SiLU (Sigmoid Linear Unit) activations for optimal training performance. However, for deployment on neuromorphic hardware (such as Intel Loihi, IBM TrueNorth, etc.), the model needs to use binary spiking activations instead.

The `convert_to_binary_spiking.py` script automatically converts all SiLU activations to binary spiking activations while preserving all model weights. Additionally, it generates a detailed architecture diagram showing the model's structure.

**Important:** By default, the architecture diagram shows all layers (including unused shortcuts and conditional layers). For an accurate representation showing only the layers and operations actually executed during inference, provide a dataset YAML file to perform a forward pass tracing. The diagram also shows both sequential and parallel operation execution paths.

## Why Convert?

**Training Phase:**
- Models are trained with SiLU activations (`mem_update(act=True)`)
- SiLU provides smooth gradients and better training convergence
- Standard PyTorch operations for efficient training

**Inference Phase:**
- Neuromorphic hardware requires binary spiking activations
- Binary spiking (`mem_update(act=False)`) is compatible with event-based processors
- Enables deployment on specialized neuromorphic chips

## What Gets Converted

The conversion process modifies three types of activation layers:

1. **Conv layers**: Changes `mem_update(act=True)` → `mem_update(act=False)`
2. **Conv_A layers**: Replaces `nn.SiLU()` → `mem_update(act=False)`
3. **mem_update modules**: Changes `act=True` → `act=False`

**Important:** Model weights remain unchanged - only activation functions are modified.

## Usage

### Basic Usage

```bash
python convert_to_binary_spiking.py --weights path/to/trained_model.pt
```

This will:
- Convert SiLU activations to binary spiking
- Save converted model as `*_binary.pt`
- Generate architecture diagram as `model_architecture.txt`

### Full Options

```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/converted_model_binary.pt \
    --diagram arch/architecture_diagram.txt \
    --device cuda \
    --data data/raddet.yaml \
    --img-size 512
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weights` | str | **Required** | Path to trained model weights (.pt file) |
| `--output` | str | `*_binary.pt` | Output path for converted model |
| `--diagram` | str | `model_architecture.txt` | Output file for architecture diagram |
| `--device` | str | `cuda` | Device to load model on (`cpu` or `cuda`) |
| `--data` | str | None | Path to dataset YAML file for forward pass tracing |
| `--img-size` | int | 640 | Input image size for forward pass tracing |
| `--no-trace` | flag | False | Disable forward pass tracing (show all modules) |

## Conversion Process

The conversion happens in three steps:

### Step 1: Convert Activations

All SiLU activations are converted to binary spiking:

```
SiLU (mem_update(act=True)) → Binary Spiking (mem_update(act=False))
```

### Step 2: Generate Architecture Diagram

A detailed architecture diagram is generated showing:
- Global parameters (time window, threshold, decay)
- Layer-by-layer decomposition
- Basic hardware building blocks
- Data flow from input to output

### Step 3: Save Converted Model

The converted model is saved with:
- All weights preserved
- Binary spiking activations
- Conversion metadata

## Output Files

### 1. Converted Model

**File:** `*_binary.pt`

PyTorch checkpoint containing:
- Converted model with binary spiking activations
- Conversion statistics
- Model metadata

**Usage:**
```python
from models.common import DetectMultiBackend

model = DetectMultiBackend('converted_model_binary.pt', device='cuda')
model.eval()
```

### 2. State Dictionary

**File:** `*_binary_state_dict.pt`

Just the model state dictionary (weights only) for easier loading.

### 3. Architecture Diagram

**File:** `model_architecture.txt` (or custom name)

Detailed text-based architecture diagram including:

#### Global Parameters
```
- Time Window (T): 5
- Threshold: 0.5
- Decay: 0.25
- Number of Classes: 11
- Input Channels: 3
- Activation: Binary Spiking (converted from SiLU)
```

#### Layer-by-Layer Decomposition

Example output showing sequential flow:
```
[Layer 0] Conv_1
  Parameters: in_channels=3, out_channels=64, kernel_size=7, stride=2
  Components:
    ├─ Snn_Conv2d →
    ├─ batch_norm_2d →
    └─ mem_update (binary spiking)
```

Example output showing parallel merge flow (residual block):
```
[Layer 1] BasicBlock_2
  Parameters: in_channels=64, out_channels=64, kernel_size=3, stride=2
  Components:
    Residual Path:
      ├─ mem_update (binary) →
      ├─ Snn_Conv2d (3×3) →
      ├─ batch_norm_2d →
      ├─ mem_update (binary) →
      ├─ Snn_Conv2d (3×3) →
      └─ batch_norm_2d1 →
    Shortcut Path: (Identity)
      (No shortcut needed: stride=1 AND channels match)
    └─ Merge: Element-wise Adder (residual)
```

**Note:** When forward pass tracing is enabled with `--data`, the diagram will only show the shortcut path if it was actually executed during the forward pass. Otherwise, it shows "(Identity)" indicating the shortcut was skipped.

#### Data Flow

```
Input Image [B, 3, H, W]
  ↓
[Temporal Replication] → [5, B, 3, H, W]
  ↓
[Backbone Layers] → Feature Extraction
  ↓
[Head Layers] → Feature Fusion
  ↓
[Detect Layer] → Object Detection
  ↓
Output [B, N_detections, 11+5]
```

## Conversion Statistics

The script provides detailed conversion statistics:

```
Conversion Statistics:
  Total modules checked: 150
  mem_update SiLU → Binary: 45
  Conv SiLU → Binary: 12
  Conv_A SiLU → Binary: 3
  Total conversions: 60
```

## Forward Pass Tracing

The script can trace which modules are actually executed during a forward pass to generate an accurate architecture diagram.

### Default Behavior

**Without `--data` flag:**
- Uses dummy data for forward pass
- May not accurately represent all execution paths
- Shows all modules, including potentially unused ones

**With `--data` flag (Recommended):**
```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --data data/raddet.yaml \
    --img-size 512
```

- Uses real test data from the dataset
- Performs actual forward pass
- Traces executed modules only
- Generates accurate architecture diagram
- Only shows layers and operations that are actually used

### Why Forward Pass Tracing Matters

The model contains conditional layers and shortcut paths that may or may not be executed depending on:
- Input dimensions
- Layer parameters (stride, channels)
- Runtime conditions

**Example:** A BasicBlock's shortcut path is only executed when:
- `stride != 1` (downsampling), OR
- Input and output channels don't match

Without forward pass tracing, the diagram shows both paths. With tracing, it only shows the path that was actually executed.

### Disable Tracing

To see all modules (including unused ones):

```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --no-trace
```

This is useful for:
- Understanding the complete model structure
- Seeing all possible execution paths
- Hardware resource planning (worst-case scenario)

### Tracing with Custom Image Size

```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --data data/raddet.yaml \
    --img-size 512
```

The image size should match your training/inference size for accurate tracing.

## Example: Converting RadDet Model

### Step 1: Convert the Model

```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/raddet_model_binary.pt \
    --diagram arch/raddet_architecture.txt \
    --device cuda \
    --data data/raddet.yaml \
    --img-size 512
```

### Step 2: Verify Conversion

Check the conversion statistics in the output:
```
✓ Conversion completed: 60 activation(s) converted to binary spiking
```

### Step 3: Use Converted Model

```python
from models.common import DetectMultiBackend

# Load converted model
model = DetectMultiBackend('arch/raddet_model_binary.pt', device='cuda')
model.eval()

# Run inference
results = model(image)
```

## Architecture Diagram Details

The generated architecture diagram provides a detailed breakdown of the model structure. **Important:** The diagram generation behavior depends on whether forward pass tracing is enabled.

### Default Behavior (Without Forward Pass Tracing)

By default, the architecture diagram shows **all layers** in the model, including:
- All shortcut paths (even if not executed)
- Conditional layers that may not be used
- All possible operations regardless of execution path

This gives a complete view of the model structure but may include unused components.

**Example:** A BasicBlock with a shortcut path will show both the residual path and shortcut path, even if the shortcut is not executed (e.g., when stride=1 and channels match).

### Accurate Representation (With Forward Pass Tracing)

To get an **accurate representation** showing only the layers and operations that are actually executed during inference, provide a dataset YAML file:

```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --data data/raddet.yaml \
    --img-size 512
```

This will:
- Perform a forward pass with real test data
- Trace which modules are actually executed
- Only include executed layers in the diagram
- Show actual shortcut paths that were used
- Provide accurate operation counts

**Result:** The diagram will only show the actual execution path, making it more accurate for hardware resource estimation and deployment planning.

### Data Flow Representation

The architecture diagram uses two types of data flow representations:

#### Sequential Flow

For layers with sequential operations, the diagram shows:

```
Components:
  ├─ Snn_Conv2d →
  ├─ batch_norm_2d →
  └─ mem_update (binary spiking)
```

This indicates operations execute one after another in sequence.

#### Parallel Merge Flow

For residual blocks (like BasicBlock), the diagram shows parallel paths that merge:

```
Components:
  Residual Path:
    ├─ mem_update (binary) →
    ├─ Snn_Conv2d (3×3) →
    ├─ batch_norm_2d →
    ├─ mem_update (binary) →
    ├─ Snn_Conv2d (3×3) →
    └─ batch_norm_2d1 →
  Shortcut Path:
    ├─ MaxPool3d (shortcut) →
    ├─ mem_update (binary) →
    ├─ Snn_Conv2d (1×1, shortcut) →
    └─ batch_norm_2d (shortcut) →
  └─ Merge: Element-wise Adder (residual)
```

This indicates:
- **Residual Path**: Main processing path (always executed)
- **Shortcut Path**: Optional path (executed when conditions are met)
- **Merge Operation**: How the paths combine (element-wise addition)

**Shortcut Conditions:**
- Shortcut is executed when: `stride != 1` OR `channels don't match`
- Shortcut is identity (skipped) when: `stride == 1` AND `channels match`

### Hardware Building Blocks

Each layer is decomposed into basic hardware building blocks:
- `Snn_Conv2d`: Spiking neural network convolution
- `batch_norm_2d`: Batch normalization
- `mem_update (binary)`: Binary spiking activation
- `MaxPool3d`: Temporal pooling
- `Element-wise Adder`: Residual connections

### Layer Information

For each layer, the diagram shows:
- Layer type and name
- Parameters (channels, kernel size, stride, etc.)
- Component breakdown with execution flow
- Data flow type (sequential or parallel merge)
- Shortcut conditions (for residual blocks)
- Whether the layer was executed (when tracing is enabled)

### Summary Statistics

- Layer type counts
- Total number of layers
- Data flow visualization
- Execution statistics (when tracing is enabled)

## Integration with Training Pipeline

Typical workflow:

1. **Train Model** (with SiLU activations):
```bash
python train.py --data data/raddet.yaml --cfg models/resnet18.yaml --epochs 300
```

2. **Convert Model** (to binary spiking):
```bash
python convert_to_binary_spiking.py \
    --weights runs/train/exp28/weights/best.pt \
    --output arch/model_binary.pt \
    --data data/raddet.yaml
```

3. **Quantize Model** (optional, for deployment):
```bash
python quantize_model.py \
    --weights arch/model_binary.pt \
    --quant-type static_int8 \
    --data data/raddet.yaml
```

4. **Deploy** on neuromorphic hardware or use for inference

## Notes and Best Practices

### Model Weights

- **Weights are preserved**: Only activation functions change
- **No retraining needed**: Conversion is lossless for activations
- **Fine-tuning optional**: May improve performance on some tasks

### Performance Considerations

- Binary spiking may have slightly different behavior than SiLU
- Consider fine-tuning if accuracy drops significantly
- Test on validation set after conversion

### Hardware Compatibility

- Binary spiking is compatible with:
  - Intel Loihi
  - IBM TrueNorth
  - Other event-based neuromorphic processors
- Check hardware-specific requirements

### Threshold Values

- Default threshold: 0.5 (hardcoded in model)
- May need adjustment for optimal performance
- Can be modified in model code if needed

## Troubleshooting

### Error: "Model file not found"

**Solution:**
- Check that the path to your model file is correct
- Use absolute path if relative path doesn't work
- Ensure the model file exists and is readable

### Error: "Error loading model"

**Solution:**
- Make sure the model was trained with this codebase
- Check that all dependencies are installed
- Verify PyTorch version compatibility

### No conversions reported

**Possible reasons:**
- Model already uses binary activations
- Model doesn't use SiLU activations
- Check model architecture to verify

**Solution:**
- Inspect the model structure
- Check conversion statistics output
- Verify model was trained with SiLU

### Forward pass tracing fails

**Solution:**
- Provide dataset YAML file with `--data`
- Use `--no-trace` to disable tracing
- Check image size matches training size
- Verify dataset path is correct

## Advanced Usage

### Custom Output Paths

```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --output custom/path/converted.pt \
    --diagram custom/path/arch.txt
```

### CPU-Only Conversion

```bash
python convert_to_binary_spiking.py \
    --weights model.pt \
    --device cpu
```

### Batch Conversion

Convert multiple models:

```bash
for model in runs/train/*/weights/best.pt; do
    python convert_to_binary_spiking.py \
        --weights "$model" \
        --output "${model%/*}/../arch/$(basename ${model%.pt})_binary.pt"
done
```

## Related Documentation

- [Architecture](architecture.md): Model architecture details and data flow
- [Training](training.md): Training guide and model preparation
- [Configuration](configuration.md): Configuration options for conversion
- [Quantization](quantization.md): Model quantization for deployment optimization
- [RadDet Use Case](usecase-raddet.md): Example conversion workflow

