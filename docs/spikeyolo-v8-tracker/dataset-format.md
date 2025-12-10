---
sidebar_position: 4
---

# Dataset Format

## eTraM Dataset Structure

The project is designed to work with the eTraM (Event-based Traffic Monitoring) dataset format.

```
eTraM/
├── HDF5/
│   ├── train_h5_6/
│   │   ├── train_night_0040_td.h5      # Event data
│   │   ├── train_night_0040_bbox.npy   # Grouped annotations (3 classes)
│   │   └── ...
│   ├── val_h5_1/
│   └── test_h5_1/
└── class annotations/
    ├── eight_class_annotations_train/
    │   ├── train_night_0040_bbox.npy   # Fine-grained annotations (8 classes)
    │   └── ...
    ├── eight_class_annotations_val/
    └── eight_class_annotations_test/
```

## Data Splits

- **Train**: ~112 sequences (mix of day/night)
- **Val**: ~23 sequences (mix of day/night)
- **Test**: ~30 sequences (mix of day/night)

## Event Data Format (.h5 files)

### HDF5 Structure

Event data is stored in HDF5 format with the following structure:

```
events/
├── x: uint16 array        # X coordinates (0-1279)
├── y: uint16 array        # Y coordinates (0-719)
├── p: int16 array         # Polarity (0=negative, 1=positive)
├── t: int64 array         # Timestamps in microseconds
├── width: int64 scalar    # Image width (1280)
└── height: int64 scalar   # Image height (720)
```

### Event Data Characteristics

- **Resolution**: 1280×720 pixels
- **Event Count**: ~17M events per sequence (5-6 seconds)
- **Temporal Resolution**: Microsecond precision
- **Polarity Distribution**: ~50/50 positive/negative events
- **Duration**: 5-6 seconds per sequence

### Example: Loading Event Data

```python
import h5py
import numpy as np

with h5py.File('train_night_0040_td.h5', 'r') as f:
    events = {
        'x': f['events/x'][:],
        'y': f['events/y'][:],
        'p': f['events/p'][:],
        't': f['events/t'][:],
        'width': f['events/width'][()],
        'height': f['events/height'][()]
    }
```

## Annotation Format (.npy files)

### Structured Array Fields

Annotations are stored as NumPy structured arrays with the following dtype:

```python
dtype = [
    ('t', '<i8'),           # Timestamp (int64)
    ('x', '<f4'),           # Top-left X coordinate (float32)
    ('y', '<f4'),           # Top-left Y coordinate (float32)
    ('w', '<f4'),           # Bounding box width (float32)
    ('h', '<f4'),           # Bounding box height (float32)
    ('class_id', '<u4'),    # Class identifier (uint32)
    ('track_id', '<u4'),    # Object tracking ID (uint32)
    ('class_confidence', '<f4')  # Detection confidence (float32)
]
```

### Bounding Box Format

- **Format**: `(x, y, w, h)` - top-left corner + width/height
- **Coordinates**: Pixel coordinates in the image space
- **Temporal Alignment**: Annotations synchronized with event timestamps

### Example: Loading Annotations

```python
import numpy as np

annotations = np.load('train_night_0040_bbox.npy')

# Access fields
timestamps = annotations['t']
bboxes = np.column_stack([
    annotations['x'],
    annotations['y'],
    annotations['w'],
    annotations['h']
])
class_ids = annotations['class_id']
track_ids = annotations['track_id']
```

## Dynamic Class Configuration

The project supports **dynamic class configuration** through `config.yaml`. Classes are defined in the `classes` list, and the number of classes is automatically detected.

### Example Configuration

```yaml
classes:
  - Pedestrian
  - Car
  - Bicycle
  - Bus
  - Motorbike
  - Truck
  - Tram
  - Wheelchair
```

### Previous Class Mappings (for reference)

- **Fine-grained 8-Class**: Pedestrian, Car, Bicycle, Bus, Motorbike, Truck, Tram, Wheelchair
- **Grouped 3-Class**: Pedestrian, Vehicle (Car, Bus, Truck, Tram), Micro-mobility (Bicycle, Motorbike, Wheelchair)

## Annotation Characteristics

- **Temporal Alignment**: Annotations synchronized with event timestamps
- **Tracking**: Each object has a unique track_id across frames
- **Confidence**: Detection confidence scores
- **Density**: ~200-300 annotations per sequence

## Sample Sequence Analysis

Example statistics from `train_night_0040`:

- **Events**: 17,428,542 events over 5.72 seconds
- **Annotations**: 212 annotations across 78 unique timestamps
- **Classes**: Pedestrian (134), Car (78)
- **Bbox Sizes**: w=294.9±148.9, h=161.7±40.7 pixels
- **Event Rate**: ~3M events/second

## Data Loading Implementation

### Key Components

1. **UltraLowMemoryLoader**: Main data loading class
2. **EventProcessor**: Event-to-frame conversion utilities
3. **eTraMDataset**: PyTorch Dataset wrapper
4. **Streaming Support**: Real-time event processing

### Dynamic Batching Process

The eTraM data loader uses a sophisticated dynamic batching system:

#### 1. Dynamic Sample Calculation

Instead of fixed samples per file, the system calculates the exact number of samples needed:

- `train_day_0001.h5`: 1,500,000 events → 150 samples (10K events each)
- `train_night_0040.h5`: 17,428,542 events → 1,743 samples (10K events each)

#### 2. Overlapping Window Sampling

Each sample uses overlapping windows to ensure complete coverage:

```python
# Use overlapping windows to ensure full coverage
overlap = self.max_events_per_sample // 4  # 25% overlap
step_size = self.max_events_per_sample - overlap  # 7,500 events step

start_idx = sample_idx * step_size
end_idx = min(start_idx + self.max_events_per_sample, total_events)
```

#### 3. Temporal Annotation Matching

Annotations are temporally matched to the specific events being processed:

```python
def _load_annotations_for_events(self, h5_file_path, events):
    """Load annotations that match the specific events temporally"""
    
    # Extract timestamps from loaded events
    event_timestamps = events[:, 2]  # Events are [x, y, t, p]
    start_time = float(event_timestamps.min())
    end_time = float(event_timestamps.max())
    
    # Add buffer for timing variations
    time_buffer = (end_time - start_time) * 0.2  # 20% buffer
    start_time -= time_buffer
    end_time += time_buffer
    
    # Filter annotations by time window
    all_annotations = self._load_all_annotations(h5_file_path)
    time_mask = (all_annotations['t'] >= start_time) & (all_annotations['t'] <= end_time)
    filtered_annotations = all_annotations[time_mask]
    
    return filtered_annotations
```

## Memory Requirements

- **Event Data**: ~200MB per sequence (HDF5 compressed)
- **Annotations**: ~50KB per sequence
- **Processing**: ~500MB RAM for real-time processing

## Camera Specifications

- **Model**: Prophesee EVK4 HD
- **Sensor**: Sony IMX636 Event-Based Vision Sensor
- **Resolution**: 1280×720 pixels
- **Dynamic Range**: >86 dB
- **Temporal Resolution**: >10,000 fps
