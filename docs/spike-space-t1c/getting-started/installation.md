---
sidebar_position: 1
---

# Installation

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11+ |
| PyTorch | 2.0 | 2.2+ |
| CUDA | -- (CPU works) | 11.8+ with a GPU |
| RAM | 8 GB | 16 GB+ |
| OS | Linux, macOS | Linux with NVIDIA GPU |

## Install from Source

```bash
git clone https://github.com/type1compute/SPIKESEG--Spiking-Neural-Network-for-Event-Based-Space-Domain-Awareness.git
cd SPIKESEG--Spiking-Neural-Network-for-Event-Based-Space-Domain-Awareness
pip install -e .
```

This installs SpikeSEG in editable mode together with all core dependencies.

## Core Dependencies

Installed automatically by `pip install -e .`:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | &ge;2.0 | Tensor operations, spiking layers |
| `numpy` | &ge;1.24 | Array manipulation |
| `scipy` | &ge;1.10 | Signal processing, sparse ops |
| `h5py` | &ge;3.8 | HDF5 dataset I/O |
| `opencv-python` | &ge;4.7 | Image processing utilities |
| `matplotlib` | &ge;3.7 | Visualization and plotting |
| `pyyaml` | &ge;6.0 | YAML configuration parsing |
| `tqdm` | &ge;4.65 | Progress bars |
| `scikit-learn` | &ge;1.2 | Clustering, metrics |

## Optional Extras

```bash
# Development tools (pytest, black, ruff, mypy)
pip install -e ".[dev]"

# Visualization extras (TensorBoard, Plotly, Seaborn)
pip install -e ".[vis]"

# Everything
pip install -e ".[all]"
```

## Verify the Installation

```bash
python -c "import spikeseg; print(spikeseg.__version__)"
# Expected output: 0.1.0
```

To confirm PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Running the Tests

```bash
pytest tests/ -v
```

Skip GPU-only tests on a CPU machine:

```bash
pytest tests/ -v -m "not cuda"
```
