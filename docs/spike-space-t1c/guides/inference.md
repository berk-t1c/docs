---
sidebar_position: 2
---

# Inference Guide

Run a trained SpikeSEG model on new data for detection or segmentation.

## CLI Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best.pt \
  --input /path/to/recording.mat \
  --output detections.json \
  --threshold 0.05 \
  --visualize
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint, -c` | *required* | Model checkpoint path |
| `--input, -i` | -- | Single `.mat` file |
| `--data-root, -d` | -- | EBSSA dataset root (batch mode) |
| `--output, -o` | `detections.json` | Output JSON path |
| `--threshold` | 0.05 | Inference threshold |
| `--visualize` | false | Save 2D detection images |
| `--visualize-3d` | false | Save 3D trajectory plots (paper style) |
| `--device` | auto | `cuda` or `cpu` |
| `--split` | `all` | Dataset split for batch mode |

### Output Format (JSON)

```json
[
  {
    "x_min": 45, "y_min": 32, "x_max": 52, "y_max": 39,
    "center_x": 48.5, "center_y": 35.5,
    "width": 7, "height": 7,
    "confidence": 0.87
  }
]
```

## Programmatic Inference

### Saliency Map (Semantic Segmentation)

```python
import torch
from spikeseg.models import SpikeSEG

model = SpikeSEG.from_paper("igarss2023", n_classes=1)
model.load_state_dict(torch.load("checkpoint.pth"))

model.reset_state()
saliency, encoder_output = model(input_events)
# saliency: (B, 1, H, W) pixel-level heat map
```

### Instance Segmentation (HULK-SMASH)

```python
from spikeseg.algorithms import HULKDecoder, group_instances_to_objects

hulk = HULKDecoder.from_encoder(model.encoder)
instances = hulk.process_to_instances(
    classification_spikes=encoder_output.classification_spikes,
    pool1_indices=encoder_output.pooling_indices.pool1_indices,
    pool2_indices=encoder_output.pooling_indices.pool2_indices,
    pool1_output_size=encoder_output.pooling_indices.pool1_output_size,
    pool2_output_size=encoder_output.pooling_indices.pool2_output_size,
    n_timesteps=10,
)
objects = group_instances_to_objects(instances, smash_threshold=0.1)
```

### Visualization

```python
from spikeseg.utils.visualization import plot_saliency_map, plot_satellite_detection

plot_saliency_map(saliency[0, 0], save_path="saliency.png")
plot_satellite_detection(events_frame, detections, save_path="detections.png")
```
