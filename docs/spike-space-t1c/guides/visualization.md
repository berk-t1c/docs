---
sidebar_position: 6
---

# Visualization Guide

SpikeSEG includes 18 matplotlib-based plot functions in `spikeseg.utils.visualization`. All accept an optional `save_path` to write a PNG/PDF file.

## Available Plots

| Function | Description |
|----------|-------------|
| `plot_learned_filters` | Grid of learned STDP convolutional filters |
| `plot_filter_evolution` | How filters change across training epochs |
| `plot_saliency_map` | Decoder output heat map |
| `plot_saliency_overlay` | Saliency overlaid on input image |
| `plot_segmentation_result` | Prediction vs ground truth side by side |
| `plot_spike_raster` | Neuron spike times over time |
| `plot_membrane_traces` | Membrane potential traces with threshold line |
| `plot_spike_activity_map` | Spatial spike counts per feature map |
| `plot_events_frame` | Single accumulated event frame |
| `plot_time_surface` | Time-surface representation of events |
| `plot_event_sequence` | Sequence of event frames across timesteps |
| `plot_convergence_metric` | $C_l$ over training steps |
| `plot_weight_distribution` | Histogram of synapse weights |
| `plot_wta_wins` | Bar chart of WTA wins per feature |
| `plot_training_metrics` | Multi-panel training dashboard |
| `plot_feature_activations` | Grid of feature map activations |
| `plot_satellite_detection` | Detections overlaid on event frame |

## Style Presets

```python
from spikeseg.utils.visualization import set_style

set_style("paper")   # Clean, publication-ready defaults
```

## Examples

### Learned Filters

```python
from spikeseg.utils.visualization import plot_learned_filters

weights = model.encoder.conv2.weight.detach().cpu()
plot_learned_filters(weights, title="Conv2 STDP Filters", save_path="filters.png")
```

### Saliency Map

```python
from spikeseg.utils.visualization import plot_saliency_map

plot_saliency_map(saliency[0, 0].cpu(), save_path="saliency.png")
```

### Convergence

```python
from spikeseg.utils.visualization import plot_convergence_metric

plot_convergence_metric(
    learner.convergence_history,
    threshold=0.01,
    save_path="convergence.png",
)
```

### Spike Raster

```python
from spikeseg.utils.visualization import plot_spike_raster

# spikes: (T, N) or (T, C, H, W) -- automatically flattened
plot_spike_raster(spikes, title="Conv2 Spike Raster", save_path="raster.png")
```

## Configuration

`VisualizationConfig` controls defaults (figsize, DPI, colormaps, font sizes, FPS):

```python
from spikeseg.utils.visualization import VisualizationConfig

cfg = VisualizationConfig(figsize=(16, 10), dpi=200, cmap_saliency="inferno")
```

## TensorBoard Integration

Plots can be logged to TensorBoard via `TensorBoardLogger.log_figure`:

```python
from spikeseg.utils.logging import TensorBoardLogger

tb = TensorBoardLogger("runs/experiment/events")
fig = plot_learned_filters(weights)
tb.log_figure("filters/conv2", fig, step=epoch)
```
