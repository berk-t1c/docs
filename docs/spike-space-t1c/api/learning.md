---
sidebar_position: 4
---

# API: Learning

STDP learning, Winner-Take-All inhibition, adaptive thresholds, and convergence tracking.

## STDPLearner

```python
class STDPLearner:
    def __init__(self, config: Optional[STDPConfig] = None)
    def initialize_weights(self, shape, device=None, dtype=torch.float32) -> Tensor
    def compute_update(self, weights, pre_spike_times, post_spike_time) -> Tensor
    def compute_batch_update(self, weights, pre_spike_times, post_spike_times, winner_mask=None) -> Tensor
    def update_weights_for_winner(
        self, weights, pre_spike_times, post_spike_time,
        winner_y, winner_x, kernel_size, stride=1, padding=0, inplace=True,
    ) -> Tensor
    def has_converged(self, weights) -> bool
    def get_convergence(self, weights) -> float
    def reset_stats(self)
    @property
    def update_count(self) -> int
    @property
    def convergence_history(self) -> List[float]
```

## STDPConfig

```python
@dataclass
class STDPConfig:
    lr_plus: float = 0.004
    lr_minus: float = 0.003
    weight_min: float = 0.0
    weight_max: float = 1.0
    variant: STDPVariant = STDPVariant.MULTIPLICATIVE  # or ADDITIVE
    convergence_threshold: float = 0.01
    weight_init_mean: float = 0.8
    weight_init_std: float = 0.01

    @classmethod
    def from_paper(cls, paper: str) -> "STDPConfig"
    # paper: "kheradpisheh2018" or "igarss2023"
```

## WTAInhibition

```python
class WTAInhibition(nn.Module):
    def __init__(
        self, config=None, n_channels=None, spatial_shape=None,
        initial_threshold=10.0, device=None,
    )
    def forward(self, spikes, membrane, pre_reset_membrane=None, threshold=None)
        -> Tuple[Tensor, Tensor]   # (filtered_spikes, winner_mask)
    def get_winner_mask(self) -> Optional[Tensor]
    def get_thresholds(self, batch_size=1) -> Optional[Tensor]
    def reset_statistics(self)
    @property
    def winner_ratio(self) -> float
```

## WTAConfig

```python
@dataclass
class WTAConfig:
    mode: WTAMode = WTAMode.GLOBAL   # GLOBAL, LOCAL, BOTH
    local_radius: int = 2
    enable_homeostasis: bool = True
    target_rate: float = 0.1
    homeostasis_lr: float = 0.001
    threshold_min: float = 1.0
    threshold_max: float = 100.0
    track_statistics: bool = True
```

## AdaptiveThreshold

```python
class AdaptiveThreshold(nn.Module):
    def __init__(
        self, n_channels, spatial_shape, initial_threshold=10.0,
        target_rate=0.1, learning_rate=0.001,
        threshold_min=1.0, threshold_max=100.0, device=None,
    )
    def update(self, spikes: Tensor)
    def get_thresholds(self, expand_batch=1) -> Tensor
    def reset_statistics(self)
    def reset_all(self, initial_threshold=10.0)
    @property
    def firing_rates(self) -> Tensor
    @property
    def mean_threshold(self) -> float
```

## ConvergenceTracker

```python
class ConvergenceTracker:
    def __init__(self, n_channels, spatial_shape, min_wins=10, delta_threshold=1e-4, device=None)
    def update(self, winner_mask, weight_deltas=None)
    @property
    def convergence_ratio(self) -> float
    @property
    def n_converged(self) -> int
    def is_converged(self, threshold=0.95) -> bool
    def reset(self)
```

## Standalone Functions

```python
def compute_convergence_metric(weights: Tensor) -> float
def has_converged(weights: Tensor, threshold: float = 0.01) -> bool
def get_first_spike_times(spikes: Tensor, no_spike_value=float('inf')) -> Tensor
def extract_receptive_field_times(pre_spike_times, post_y, post_x, kernel_size, stride=1, padding=0) -> Tensor
def compute_stdp_update(weights, pre_spike_times, post_spike_time, lr_plus, lr_minus, variant=...) -> Tensor
def find_wta_winner(output_spikes, potentials=None) -> Tuple[Optional[int], Optional[Tuple[int,int]], Optional[float]]
def apply_lateral_inhibition(potentials, winner_map, winner_y, winner_x, inplace=True) -> Tensor
```
