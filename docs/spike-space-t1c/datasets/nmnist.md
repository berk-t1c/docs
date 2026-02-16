---
sidebar_position: 2
---

# N-MNIST Dataset

**Neuromorphic MNIST (N-MNIST)** [8] is a spiking version of the classic MNIST handwritten digit dataset, created by displaying each MNIST image on a monitor and recording the output of a neuromorphic sensor undergoing saccadic motion.

## Overview

| Property | Value |
|----------|-------|
| Resolution | 34 x 34 pixels |
| Classes | 10 digits (0--9) |
| Training samples | 60,000 |
| Test samples | 10,000 |
| File format | Binary event files |
| Event fields | x, y, polarity, timestamp |

## Usage

```python
from spikeseg.data import NMNISTDataset

train_ds = NMNISTDataset(
    root="/path/to/N-MNIST",
    train=True,
    n_timesteps=10,
    height=34,
    width=34,
    normalize=True,
    polarity_channels=True,
)

for events, label in train_ds:
    # events: (T, 2, 34, 34) voxel grid
    # label: int 0-9
    pass
```

## Purpose in SpikeSEG

N-MNIST serves as a **benchmarking** dataset. The architecture and STDP training pipeline were originally developed and validated on N-MNIST [1] before being applied to the EBSSA space domain dataset [4].

## Citation

> G. Orchard, A. Jayawant, G. K. Cohen, and N. Thakor, "Converting static image datasets to spiking neuromorphic datasets using saccades," *Frontiers in Neuroscience*, vol. 9, p. 437, 2015.
