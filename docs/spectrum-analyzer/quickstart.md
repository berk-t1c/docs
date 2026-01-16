---
sidebar_position: 2
---

# Quick Start

Get started with Spectrum Analyzer in minutes.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0+ with CUDA support

### Step 1: Clone Repository

```bash
git clone https://github.com/type1compute/Spectrum-Analyzer.git
cd Spectrum-Analyzer
```

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install PyTorch

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8+
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1+
pip install torch>=2.0.0+cu121 torchvision>=0.15.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

## Using Pre-trained Model

### Download Pre-trained Model

Pre-trained models are available at: [Google Drive](https://drive.google.com/drive/folders/1z-cGQhVtElTe6ZCqa5bihQ8guwyl9kaG?usp=sharing)

### Run Detection

```bash
python detect.py \
    --weights path/to/best.pt \
    --source path/to/images \
    --imgsz 512 \
    --conf-thres 0.25 \
    --save-img true
```

Results will be saved to `runs/detect/exp/`

## Training on Your Dataset

### Step 1: Prepare Dataset

Organize your dataset:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Step 2: Create Dataset Config

Create `data/your_dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 11  # number of classes
names: ['class1', 'class2', ...]
```

### Step 3: Train Model

```bash
python train.py \
    --data data/your_dataset.yaml \
    --cfg models/resnet18.yaml \
    --imgsz 512 \
    --epochs 300 \
    --batch-size 64 \
    --device 0
```

For detailed training instructions, see the [Training Guide](training.md).

### Step 4: Validate Model

```bash
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/your_dataset.yaml \
    --task test \
    --imgsz 512
```

For more validation options, see [Configuration](configuration.md). For detailed training instructions, see the [Training Guide](training.md).

## Next Steps

- Learn about [Architecture](architecture.md) - Understand how the system works
- Explore [Configuration Options](configuration.md) - Configure for your use case
- Read [Training Guide](training.md) - Train on your dataset
- Check [Detection Guide](detection.md) - Run inference
- See [RadDet Use Case](usecase-raddet.md) - Example demonstration
- Convert models: [Model Conversion](model-conversion.md) - Prepare for deployment
- Optimize models: [Model Quantization](quantization.md) - Optimize for deployment

