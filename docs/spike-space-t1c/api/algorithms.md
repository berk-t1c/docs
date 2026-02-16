---
sidebar_position: 5
---

# API: Algorithms

HULK instance-wise decoding, ASH hashing, SMASH scoring, and instance/object types.

## HULKDecoder

```python
class HULKDecoder(nn.Module):
    def __init__(
        self,
        trans_conv1_weight: Tensor,
        trans_conv2_weight: Tensor,
        conv3_weight: Tensor,
        kernel_sizes: Tuple[int, int, int] = (5, 5, 7),
        pool_kernel_size: int = 2,
    )
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> "HULKDecoder"

    def unravel_spike(
        self, spike_location, timestep, pool1_indices, pool2_indices,
        pool1_output_size, pool2_output_size, class_id=0, threshold=0.5,
    ) -> HULKResult

    def unravel_all_spikes(
        self, classification_spikes, pool1_indices, pool2_indices,
        pool1_output_size, pool2_output_size, n_timesteps, threshold=0.5,
    ) -> List[HULKResult]

    def process_to_instances(
        self, classification_spikes, pool1_indices, pool2_indices,
        pool1_output_size, pool2_output_size, n_timesteps, threshold=0.5,
    ) -> List[Instance]
```

## HULKResult

```python
@dataclass
class HULKResult:
    spike_location: Tuple[int, int]
    pixel_mask: Tensor
    layer_activities: Dict[str, LayerSpikeActivity]
    ash: Optional[ActiveSpikeHash] = None
    bbox: Optional[BoundingBox] = None

    def compute_ash(self, n_features, n_timesteps) -> ActiveSpikeHash
    def compute_bbox(self) -> Optional[BoundingBox]
    def to_instance(self, instance_id, n_features, n_timesteps, class_id=0) -> Instance
```

## ActiveSpikeHash

Binary feature-time matrix for similarity comparison.

```python
@dataclass
class ActiveSpikeHash:
    hash_matrix: Tensor      # (n_features, n_timesteps), binary
    n_features: int
    n_timesteps: int

    @classmethod
    def from_spike_activity(cls, spike_times, n_features, n_timesteps, device=None) -> "ActiveSpikeHash"
    @classmethod
    def from_spike_tensor(cls, spikes: Tensor) -> "ActiveSpikeHash"

    def similarity(self, other: "ActiveSpikeHash") -> float   # Jaccard
    @property
    def sparsity(self) -> float
    @property
    def n_active(self) -> int
```

## BoundingBox

```python
@dataclass
class BoundingBox:
    x_min: int; y_min: int; x_max: int; y_max: int

    @classmethod
    def from_mask(cls, mask: Tensor) -> Optional["BoundingBox"]

    @property
    def width(self) -> int
    @property
    def height(self) -> int
    @property
    def area(self) -> int
    @property
    def center(self) -> Tuple[float, float]

    def iou(self, other: "BoundingBox") -> float
    def to_xywh(self) -> Tuple[int, int, int, int]
    def to_xyxy(self) -> Tuple[int, int, int, int]
```

## Instance and Object

```python
@dataclass
class Instance:
    instance_id: int
    ash: ActiveSpikeHash
    bbox: BoundingBox
    class_id: int = 0
    mask: Optional[Tensor] = None
    spike_location: Optional[Tuple[int, int]] = None

    def smash_score(self, other: "Instance") -> float

@dataclass
class Object:
    object_id: int
    instances: List[Instance]
    combined_ash: Optional[ActiveSpikeHash] = None
    combined_bbox: Optional[BoundingBox] = None

    def add_instance(self, instance: Instance)
    @property
    def n_instances(self) -> int
```

## Scoring and Grouping

```python
def compute_smash_score(ash1, bbox1, ash2, bbox2) -> float
def group_instances_to_objects(instances: List[Instance], smash_threshold: float = 0.0) -> List[Object]
def match_objects_across_sequences(current_objects, previous_objects, similarity_threshold=0.0) -> Dict[int, int]
```
