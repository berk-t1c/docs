---
sidebar_position: 3
---

# Installation & Setup

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Compatible GPU with CUDA support (recommended)
- **Operating System**: Linux, macOS, or Windows

## Clone the Repository

```bash
git clone https://github.com/type1compute/SpikeYoloV8-Tracker.git
cd SpikeYoloV8-Tracker
```

## Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Key Dependencies

The project requires:

- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation base
- **NumPy**: Numerical computations
- **H5py**: HDF5 file handling
- **Other dependencies**: See `requirements.txt` for complete list

## Verify Installation

Check GPU availability (if using CUDA):

```bash
nvidia-smi
```

## Project Structure Setup

Ensure your project structure matches the expected layout:

```
SpikeYoloV8-Tracker/
├── config/
│   └── config.yaml
├── src/
├── scripts/
├── HDF5/              # Place your event data here
└── class annotations/ # Place your annotations here
```

## Configuration

Create or modify `config/config.yaml` with your settings. See the [Configuration Guide](./configuration) for details.

## Next Steps

- [Quick Start Guide](./quick-start)
- [Dataset Format](./dataset-format)
- [Configuration Guide](./configuration)
