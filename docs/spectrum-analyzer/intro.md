---
sidebar_position: 1
---

# Introduction

Spectrum Analyzer is a deep learning-based tool for detecting and classifying signals from spectrogram images. Built on the EMS-YOLO codebase, it provides a flexible framework for spectrum analysis with configurable parameters for different image sizes and use cases.

**Repository:** [GitHub](https://github.com/type1compute/Spectrum-Analyzer)  
**Pre-trained Models:** [Google Drive](https://drive.google.com/drive/folders/1z-cGQhVtElTe6ZCqa5bihQ8guwyl9kaG?usp=sharing)

## What is Spectrum Analyzer?

Spectrum Analyzer uses an enhanced YOLO (You Only Look Once) object detection framework to identify and classify various signal types in spectrogram images. The system is designed to be general-purpose, allowing users to configure it for different image sizes, model architectures, and detection scenarios.

## Key Features

- **Multi-class Signal Detection**: Detects and classifies multiple signal types simultaneously
- **Configurable Architecture**: Support for different model backbones (ResNet-10, ResNet-18, ResNet-34)
- **Flexible Image Sizes**: Configurable input image resolution (default: 512×512, configurable)
- **Time Window Processing**: Temporal information integration using 5-frame time windows
- **High Precision**: Achieves up to 100% precision at high confidence thresholds
- **Real-time Inference**: Fast detection on spectrogram images
- **Model Quantization**: Support for model quantization for deployment

## Use Cases

Spectrum Analyzer can be used for:

- **Radar Signal Detection**: Identifying various radar signal types in spectrograms
- **Communication Signal Analysis**: Detecting and classifying communication signals
- **Spectrum Monitoring**: Real-time spectrum analysis and signal detection
- **Research Applications**: Custom signal detection tasks with configurable parameters

## Demonstration: RadDet Dataset

This codebase has been demonstrated using the **RadDet** dataset, which contains 11 different radar signal types:

- Rect, Barker, Frank, P1-P4, Px, ZadoffChu, LFM, FMCW

The model achieves:
- **mAP@0.5**: 0.39
- **Precision**: 0.39
- **Recall**: 0.55-0.58
- **100% precision** at confidence threshold 0.932

![Training Results](/img/spectrum_analyzer/training-results.png)

## Architecture Overview

The system uses a modified YOLO architecture with:

- **Backbone**: ResNet-based feature extractor (configurable)
- **Detection Head**: Multi-scale YOLO detection layers
- **Time Window**: 5-frame temporal processing
- **Output**: Bounding boxes, confidence scores, and class predictions

## Getting Started

To get started with Spectrum Analyzer:

1. **Install Dependencies**: Follow the installation guide
2. **Configure Dataset**: Set up your dataset configuration file
3. **Train or Use Pre-trained Model**: Train on your data or use provided models
4. **Run Detection**: Use the detection script for inference

For quick setup, see the [Quick Start Guide](quickstart.md). For detailed documentation keep following this guide and read through all the tabs.

## Next Steps

- Learn about the [Architecture](architecture.md)
- Understand [Configuration Options](configuration.md)
- See the [RadDet Use Case](usecase-raddet.md)
- Explore [Training Parameters](training.md)
- Convert models for deployment: [Model Conversion](model-conversion.md)
- Optimize models: [Model Quantization](quantization.md)

