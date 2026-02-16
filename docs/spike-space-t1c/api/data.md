---
sidebar_position: 6
---

# API: Data

Dataset loaders, event representations, and preprocessing pipelines.

## Datasets

### EBSSADataset

```python
class EBSSADataset(EventDataset):
    SENSORS = {'ATIS': {'height': 240, 'width': 304}, 'DAVIS': {'height': 180, 'width': 240}}

    def __init__(
        self, root, split="train", sensor="DAVIS",
        n_timesteps=10, height=None, width=None,
        normalize=True, polarity_channels=True,
        use_labels=True, include_unlabelled=False,
        train_ratio=0.8, transform=None, augmentation=None,
        max_samples=None, windows_per_recording=1, window_overlap=0.5,
    )
    def __len__(self) -> int
    def __getitem__(self, index) -> Tuple[Tensor, Any]
    def get_recording_info(self, index) -> Dict[str, Any]
```

### NMNISTDataset

```python
class NMNISTDataset(EventDataset):
    def __init__(
        self, root, train=True, n_timesteps=10,
        height=34, width=34, normalize=True,
        polarity_channels=True, transform=None,
        augmentation=None, max_samples=None,
    )
```

### SyntheticEventDataset

```python
class SyntheticEventDataset(EventDataset):
    def __init__(
        self, n_samples=100, n_events_per_sample=1000,
        height=128, width=128, n_classes=2,
        n_timesteps=10, normalize=True, polarity_channels=True, seed=None,
    )
```

## Event Representations

### EventData

```python
@dataclass
class EventData:
    x: np.ndarray; y: np.ndarray; p: np.ndarray; t: np.ndarray
    height: int; width: int

    @property
    def n_events(self) -> int
    @property
    def duration(self) -> float
    @property
    def event_rate(self) -> float
    def filter_by_time(self, t_start, t_end) -> "EventData"
    def filter_by_region(self, x_min, x_max, y_min, y_max) -> "EventData"
```

### Conversion Functions

```python
def events_to_voxel_grid(events, n_timesteps, height=None, width=None, normalize=True, polarity_channels=True) -> Tensor
def events_to_frame(events, height=None, width=None, accumulate=True, polarity_channels=True) -> Tensor
def events_to_time_surface(events, height=None, width=None, tau=0.1, polarity_channels=True) -> Tensor
```

### I/O

```python
def load_events_mat(filepath, height=180, width=240) -> EventData
def load_events_h5(filepath, recording_name=None, height=180, width=240) -> EventData
def load_events_npy(filepath, height=180, width=240) -> EventData
def load_events(filepath, height=180, width=240) -> EventData       # auto-detect format
def load_labels_mat(filepath) -> Dict[str, Any]
```

## Preprocessing

### SpikeSEGPreprocessor

For event camera streams (EBSSA).

```python
class SpikeSEGPreprocessor(nn.Module):
    def __init__(
        self, height=180, width=240, n_channels=2,
        n_event_steps=10, n_propagation_steps=10,
        use_lif_buffer=True, lif_tau=10.0,
        use_adaptive_threshold=True, base_threshold=1.0, target_rate=0.1,
    )
    def forward(self, x, y, p, t) -> Tuple[Tensor, Tensor]
    def reset(self)
```

### SpykeTorchPreprocessor

For static images converted to spikes (N-MNIST).

```python
class SpykeTorchPreprocessor(nn.Module):
    def __init__(
        self, filter_type="dog",
        dog_sizes=None, n_orientations=4, kernel_size=7,
        normalization_radius=4, n_timesteps=15, threshold=0.01,
        use_lateral_inhibition=False, use_pointwise_inhibition=True,
    )
    def forward(self, x: Tensor) -> Tensor
```

### Event Filtering

```python
def filter_refractory_events(x, y, p, t, refractory_period=1000.0) -> Tuple[...]
def filter_isolated_events(x, y, p, t, spatial_radius=1, temporal_window=10000.0, min_neighbors=1) -> Tuple[...]
```

### DoG and Gabor Filters

```python
class DoGFilter(nn.Module):
    def __init__(self, sizes=None, sigma_ratio=1.6, threshold=0.0, padding='same')
    def forward(self, x) -> Tensor

class GaborFilterBank(nn.Module):
    def __init__(self, n_orientations=4, n_scales=1, kernel_size=7, use_abs=True, threshold=0.0)
    def forward(self, x) -> Tensor
```

### Augmentation

```python
class EventAugmentation:
    def __init__(self, flip_horizontal=False, flip_vertical=False, flip_polarity=False,
                 random_crop=None, noise_rate=0.0, drop_rate=0.0)
    def __call__(self, events: EventData) -> EventData

class SpikeAugmentation(nn.Module):
    def __init__(self, flip_horizontal=True, flip_vertical=False, flip_temporal=False,
                 jitter_time=0, dropout_prob=0.0, noise_prob=0.0)
    def forward(self, x: Tensor) -> Tensor
```

### DataLoader Helper

```python
def create_dataloader(dataset, batch_size=1, shuffle=True, num_workers=4,
                      pin_memory=True, drop_last=False, collate_fn=None) -> DataLoader
def get_dataset(name: str, root: str, split="train", **kwargs) -> EventDataset
```
