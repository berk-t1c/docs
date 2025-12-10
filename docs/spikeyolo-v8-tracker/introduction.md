---
sidebar_position: 1
---

# SpikeYoloV8-Tracker Introduction

**SpikeYoloV8-Tracker** is a complete end-to-end pipeline for real-time object detection and tracking using event camera data. The architecture is adapted from the BICLab SpikeYOLO ECCV 2024 implementation.

## Key Features

- **BICLab ECCV 2024 Implementation**: Original SpikeYOLO with I-LIF spiking neurons
- **Configurable-Class Detection**: Detects multiple object classes dynamically
- **Object Tracking**: Tracks objects through time using Hungarian-algorithm based ByteTracker
- **Event Processing**: Converts event data to spike trains for SNN processing
- **End-to-End Pipeline**: Contains highly configurable training and testing pipeline

## What is Event-Based Vision?

Event cameras (also known as neuromorphic cameras) are bio-inspired sensors that capture changes in brightness at each pixel independently, rather than capturing full frames at fixed intervals. This provides:

- **High Temporal Resolution**: Microsecond-level precision
- **High Dynamic Range**: >86 dB
- **Low Latency**: Event-by-event processing
- **Energy Efficiency**: Only processes changes in the scene

## Use Cases

SpikeYoloV8-Tracker is ideal for:

- **Traffic Monitoring**: Real-time detection and tracking of vehicles, pedestrians, and other traffic participants
- **Surveillance Systems**: Low-power, high-speed object tracking
- **Autonomous Systems**: Event-based perception for robotics and autonomous vehicles
- **Research**: Spiking neural networks for event-based vision

## Project Repository

The source code is available at: [https://github.com/type1compute/SpikeYoloV8-Tracker](https://github.com/type1compute/SpikeYoloV8-Tracker)

## Next Steps

- [Installation Guide](./installation)
- [Architecture Overview](./architecture)
- [Quick Start Guide](./quick-start)
