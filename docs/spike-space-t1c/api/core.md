---
sidebar_position: 2
---

# API: Core

Neuron dynamics, spiking layers, and stateless functional operations.

## spikeseg.core.neurons

### BaseNeuron (ABC)

```python
class BaseNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0)
    def forward(self, input_current, membrane, has_fired=None, dynamic_threshold=None)
        -> Tuple[Tensor, Tensor, Tensor]   # (spikes, membrane, pre_reset_membrane)
    def reset_state(self, shape, device) -> Tensor
```

### IFNeuron

Integrate-and-Fire neuron. $V(t) = V(t-1) + I(t)$.

```python
class IFNeuron(BaseNeuron):
    def __init__(self, threshold: float = 1.0)
```

### LIFNeuron

Leaky Integrate-and-Fire neuron with configurable leak mode.

```python
class LIFNeuron(BaseNeuron):
    def __init__(
        self,
        threshold: float = 1.0,
        leak_factor: float = 0.0,
        leak_mode: Literal["subtractive", "multiplicative"] = "subtractive",
    )
```

### Factory

```python
def create_neuron(
    neuron_type: str,       # "if" or "lif"
    threshold: float,
    leak_factor: float = 0.0,
    leak_mode: str = "subtractive",
) -> BaseNeuron
```

---

## spikeseg.core.layers

### SpikingConv2d

Convolutional layer with integrated spiking neuron.

```python
class SpikingConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[int, str] = 0,
        neuron_type: str = "lif",
        threshold: float = 1.0,
        leak_factor: float = 0.0,
        leak_mode: str = "subtractive",
        learnable: bool = True,
        weight_init_mean: float = 0.8,
        weight_init_std: float = 0.01,
    )
    def forward(self, x, n_timesteps=1, membrane=None, return_all_timesteps=False)
    def reset(self)
```

### SpikingPool2d

Max pooling with index preservation.

```python
class SpikingPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: int = 0,
        pool_mode: Literal["spike_count", "first_spike"] = "spike_count",
    )
    def forward(self, x, spike_times=None) -> Tuple[Tensor, Tensor]
```

### SpikingUnpool2d

```python
class SpikingUnpool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0)
    def forward(self, x, indices, output_size=None) -> Tensor
```

### SpikingTransposedConv2d

```python
class SpikingTransposedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0)
    def tie_weights(self, encoder_conv: SpikingConv2d)
    def forward(self, x) -> Tensor
```

---

## spikeseg.core.functional

### Spike Generation

```python
def spike_fn(membrane: Tensor, threshold: float) -> Tensor
def soft_spike_fn(membrane: Tensor, threshold: float, temperature: float = 1.0) -> Tensor
```

### Membrane Steps

```python
def if_step(membrane, input_current, threshold, reset_value=0.0) -> (Tensor, Tensor)
def lif_step_subtractive(membrane, input_current, threshold, leak, reset_value=0.0) -> (Tensor, Tensor)
def lif_step_multiplicative(membrane, input_current, threshold, beta, reset_value=0.0) -> (Tensor, Tensor)
def lif_step(membrane, input_current, threshold, leak_factor, leak_mode="subtractive", reset_value=0.0) -> (Tensor, Tensor)
```

### Filter Creation

```python
def create_gaussian_kernel(size: int, sigma: float, normalize: bool = True) -> Tensor
def create_dog_filters(size: int = 7, sigma_center: float = 1.0, sigma_surround: float = 2.0) -> Tensor
def create_gabor_filters(size: int = 5, sigma: float = 1.0, frequency: float = 0.5, n_orientations: int = 4) -> Tensor
```

### Temporal Encoding

```python
def intensity_to_latency(intensity: Tensor, max_time: float = 1.0, epsilon: float = 1e-6) -> Tensor
def latency_to_spikes(latency: Tensor, n_timesteps: int, max_time: float = 1.0) -> Tensor
def encode_image_to_spikes(image: Tensor, n_timesteps: int, dog_size: int = 7, ...) -> Tensor
```

### Winner-Take-All

```python
def wta_global(spikes, membrane, pre_reset_membrane=None) -> (Tensor, Tensor)
def wta_local(spikes, membrane, radius: int = 2) -> (Tensor, Tensor)
```

### Utilities

```python
def compute_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1) -> int
def count_spikes(spikes: Tensor, dim=None) -> Tensor
```
