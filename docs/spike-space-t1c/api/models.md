---
sidebar_position: 3
---

# API: Models

Encoder, decoder, and the combined SpikeSEG model.

## SpikeSEG

Complete model with encoder and lazy-initialized decoder.

```python
class SpikeSEG(nn.Module):
    def __init__(self, config=None, decoder_config=None, device=None)

    @classmethod
    def from_paper(cls, paper: str, n_classes: int = 1, decoder_config=None, device=None) -> "SpikeSEG"

    @classmethod
    def from_config(
        cls,
        conv1_channels=4, conv2_channels=36, n_classes=1,
        kernel_sizes=(5, 5, 7),
        thresholds=(10.0, 10.0, 10.0),
        leaks=(9.0, 1.0, 0.0),
        device=None,
    ) -> "SpikeSEG"

    def reset_state(self)
    def encode(self, x, layer_thresholds=None) -> EncoderOutput
    def decode(self, encoder_output) -> Tensor
    def forward(self, x, return_encoder_output=True) -> Union[Tensor, Tuple[Tensor, EncoderOutput]]
    def get_layer_weights(self) -> dict
    def freeze_layer(self, layer_name: str)
    def unfreeze_layer(self, layer_name: str)
    @property
    def n_classes(self) -> int
    @property
    def decoder(self) -> SpikeSEGDecoder  # lazy init
```

## SpikeSEGEncoder

```python
class SpikeSEGEncoder(nn.Module):
    def __init__(self, config: Optional[EncoderConfig] = None)
    def reset_state(self)
    def forward_single_timestep(self, x, layer_thresholds=None) -> Tuple[Tensor, Dict[str, Tensor]]
    def forward(self, x, n_timesteps=None, reset_state=True, layer_thresholds=None) -> EncoderOutput
    def get_pooling_indices(self) -> PoolingIndices
    def get_feature_counts(self) -> Dict[str, int]
    @property
    def n_classes(self) -> int
    @property
    def kernel_sizes(self) -> Tuple[int, int, int]
```

## SpikeSEGDecoder

```python
class SpikeSEGDecoder(nn.Module):
    def __init__(
        self, n_classes, conv2_channels, conv1_channels, input_channels=1,
        kernel_sizes=(5, 5, 7), pool_kernel_size=2,
        config=None, encoder_weights=None,
    )
    @classmethod
    def from_encoder(cls, encoder, config=None) -> "SpikeSEGDecoder"
    def reset_state(self)
    def forward(self, classification_spikes, pool1_indices, pool2_indices,
                pool1_output_size, pool2_output_size) -> Tensor
    def decode_single_spike(self, spike_location, class_id, batch_size, class_spatial_shape,
                            pool1_indices, pool2_indices, pool1_output_size, pool2_output_size,
                            device=None) -> Tensor
```

## Configuration Dataclasses

### EncoderConfig

```python
@dataclass
class EncoderConfig:
    input_channels: int = 1
    conv1: LayerConfig
    conv2: LayerConfig
    conv3: LayerConfig
    pool1_kernel_size: int = 2
    pool1_stride: int = 2
    pool2_kernel_size: int = 2
    pool2_stride: int = 2
    use_wta: bool = True
    wta_mode: str = "both"
    store_all_spikes: bool = True
    store_membranes: bool = False

    @classmethod
    def from_paper(cls, paper: str, n_classes: int = 1) -> "EncoderConfig"
    def with_n_classes(self, n_classes: int) -> "EncoderConfig"
```

### LayerConfig

```python
@dataclass
class LayerConfig:
    out_channels: int
    kernel_size: int
    threshold: float = 10.0
    leak: float = 0.0
    leak_mode: str = "subtractive"
```

### DecoderConfig

```python
@dataclass
class DecoderConfig:
    use_tied_weights: bool = True
    use_spiking: bool = True
    threshold: float = 1.0
    leak: float = 0.0
    use_delay_connections: bool = True
    delay_steps: int = 1
```

### EncoderOutput

```python
@dataclass
class EncoderOutput:
    classification_spikes: Tensor
    pooling_indices: PoolingIndices
    layer_spikes: Dict[str, Tensor]
    layer_membranes: Optional[Dict[str, Tensor]] = None
    layer_spike_times: Optional[Dict[str, Tensor]] = None

    @property
    def has_spikes(self) -> bool
    @property
    def n_classification_spikes(self) -> int
```

### PoolingIndices

```python
class PoolingIndices(NamedTuple):
    pool1_indices: Tensor
    pool2_indices: Tensor
    pool1_output_size: Tuple[int, int, int, int]
    pool2_output_size: Tuple[int, int, int, int]
```
