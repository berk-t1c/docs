---
sidebar_position: 1
---

# EBSSA Dataset

The **Event-Based Space Situational Awareness (EBSSA)** dataset [7] contains neuromorphic event camera recordings of resident space objects (satellites, rocket bodies), planets, and stars.

## Overview

| Property | Value |
|----------|-------|
| Sensors | ATIS (304 x 240) and DAVIS240C (240 x 180) |
| Labelled recordings | 84 |
| Unlabelled recordings | 153 |
| File format | MATLAB `.mat` (per recording) or HDF5 (combined) |
| Event fields | x, y, polarity, timestamp |
| Labels | Bounding box trajectories (expert annotated) |
| Source | Western Sydney University, International Centre for Neuromorphic Systems |

## Sample Recording

The following video shows a raw EBSSA recording of a satellite (SL-8 rocket body, NORAD 21938) tracked against a star field:

<div class="video-container">
  <video controls playsinline preload="metadata" style={{ width: '100%', height: 'auto' }}>
    <source src="/video/spike-space-t1c/ebssa_satellite_recording.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>

*Recording 20170214-21-15, SL8RB (NORAD 21938). The satellite appears as a faint streak moving across the field of view while stars produce stationary event clusters.*

## Expert Labels

This video shows the EBSSA expert label overlay, demonstrating the ground-truth bounding box annotations used for evaluation:

<div class="video-container">
  <video controls width="100%">
    <source src="/video/spike-space-t1c/ebssa_expert_labels.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video>
</div>

*Expert label data showing bounding box annotations tracking satellites across event camera recordings.*

## Directory Layout

```text
EBSSA/
├── Labelled Data/
│   ├── 20170214-20-58_22285_SL-16RB_labelled.mat
│   ├── archenar_leos_11_33_atis_td_labelled.mat
│   ├── archenar_leos_11_33_davis_td_labelled.mat
│   ├── ...                    (84 recordings)
│   └── HDF5_Format/
│       ├── plot_trajectory.py
│       └── Readme.txt
├── Unlabelled Data/
│   └── ...                    (153 recordings)
├── converted/                 (optional: pre-converted .h5 + .npy)
│   ├── train_h5_1/
│   └── val_h5_1/
└── Readme.txt
```

## Usage

```python
from spikeseg.data import EBSSADataset

dataset = EBSSADataset(
    root="/path/to/EBSSA",
    split="train",
    sensor="all",         # "ATIS", "DAVIS", or "all"
    n_timesteps=10,
    height=128,
    width=128,
    polarity_channels=True,   # 2 channels (ON/OFF)
    train_ratio=0.9,
)

for events, labels in dataset:
    # events: (T, C, H, W) voxel grid
    # labels: bounding box mask or dict
    pass
```

## Configuration

```yaml
data:
  dataset: "ebssa"
  data_root: "/path/to/EBSSA"
  sensor: "all"
  n_timesteps: 10
  input_height: 128
  input_width: 128
  input_channels: 2
  windows_per_recording: 1
```

## Citation

> S. Afshar, A. P. Nicholson, A. van Schaik, and G. Cohen, "Event-based object detection and tracking for space situational awareness," *IEEE Sensors Journal*, vol. 20, no. 24, pp. 15117--15132, 2020.
