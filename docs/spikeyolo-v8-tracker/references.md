---
sidebar_position: 10
---

# Research References

## BICLab SpikeYOLO (ECCV 2024)

### Paper

**Title**: "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection"

**Authors**: 
- Xinhao Luo
- Man Yao
- Yuhong Chou
- Bo Xu
- Guoqi Li

**Institution**: BICLab, Institute of Automation, Chinese Academy of Sciences

**Conference**: ECCV 2024

### Key Contributions

- Integer-valued training for spiking neural networks
- Spike-driven inference for energy efficiency
- Adaptation of YOLO architecture for SNNs
- I-LIF (Integer-valued LIF) neuron implementation

### Repository

The original BICLab SpikeYOLO implementation serves as the foundation for this project.

## eTraM Dataset

### Dataset Information

**Name**: Event-based Traffic Monitoring Dataset

**Characteristics**:
- **Resolution**: 1280×720 pixels
- **Classes**: 8 traffic participant classes
- **Format**: HDF5 event files with NumPy annotations
- **Camera**: Prophesee EVK4 HD (Sony IMX636 sensor)

### Classes

1. Pedestrian
2. Car
3. Bicycle
4. Bus
5. Motorbike
6. Truck
7. Tram
8. Wheelchair

## ByteTracker

### Tracking Algorithm

ByteTracker is used for object tracking in this project:

- **Algorithm**: Hungarian algorithm-based tracking
- **Features**: Handles occlusions and re-identifications
- **Association**: Uses detection features for object association

## Related Work

### Event-Based Vision

Event cameras capture changes in brightness at each pixel independently, providing:

- High temporal resolution (microsecond precision)
- High dynamic range (>86 dB)
- Low latency (event-by-event processing)
- Energy efficiency (only processes changes)

### Spiking Neural Networks

SNNs are biologically-inspired neural networks that:

- Process information through spikes
- Operate on discrete time steps
- Offer energy efficiency benefits
- Enable temporal processing

### Object Detection

YOLO (You Only Look Once) architecture adapted for:

- Event-based input
- Spiking neural network processing
- Real-time object detection
- Multi-class detection

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{spikeyolo2024,
  title={Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection},
  author={Luo, Xinhao and Yao, Man and Chou, Yuhong and Xu, Bo and Li, Guoqi},
  journal={ECCV},
  year={2024}
}
```

## Additional Resources

### Event Camera Resources

- Prophesee: [https://www.prophesee.ai/](https://www.prophesee.ai/)
- Event Camera Datasets: Various datasets available for event-based vision research

### Spiking Neural Networks

- SNN Research: Active research area in neuromorphic computing
- Energy Efficiency: Key advantage of SNNs for edge devices

### Object Detection

- YOLO: Popular real-time object detection framework
- Detection Metrics: mAP, precision, recall for evaluation

## Acknowledgments

This project builds upon:

- BICLab SpikeYOLO implementation
- eTraM dataset
- ByteTracker algorithm
- YOLO architecture

## License

Please refer to the repository for license information.
