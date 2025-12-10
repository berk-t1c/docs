---
sidebar_position: 7
---

# Evaluation Guide

## Overview

SpikeYoloV8-Tracker provides comprehensive evaluation tools to assess model performance on detection and tracking tasks.

## Evaluation Script

### Main Evaluation Script

**Location**: `scripts/evaluation/targeted_model_evaluation.py`

Run evaluation on a trained model:

```bash
python scripts/evaluation/targeted_model_evaluation.py \
    --checkpoint_path /path/to/checkpoint.pt
```

### Command-Line Arguments

- `--checkpoint_path`: Path to model checkpoint file (required)
- `--device`: Device to use for evaluation (default: "cuda")
- `--batch_size`: Batch size for evaluation (default: from config)
- `--output_dir`: Directory to save evaluation results (optional)

## Evaluation Metrics

### Detection Metrics

The evaluation script computes standard object detection metrics:

- **mAP (mean Average Precision)**: Overall detection accuracy
- **mAP@0.5**: mAP at IoU threshold 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds 0.5 to 0.95
- **Per-Class AP**: Average precision for each class
- **Precision**: Ratio of true positives to all detections
- **Recall**: Ratio of true positives to all ground truth objects

### Tracking Metrics

For tracking evaluation:

- **MOTA (Multiple Object Tracking Accuracy)**: Overall tracking accuracy
- **MOTP (Multiple Object Tracking Precision)**: Tracking precision
- **ID Switches**: Number of identity switches
- **Track Fragmentation**: Number of track fragments

## Evaluation Process

### 1. Model Loading

The evaluation script loads the trained model from checkpoint:

```python
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 2. Data Loading

Evaluation uses the same data loading pipeline as training:

- Loads test/validation sequences
- Processes events in temporal windows
- Matches annotations temporally

### 3. Inference

Model runs in evaluation mode:

- No gradient computation
- Batch processing
- Temporal-aware predictions

### 4. Metric Computation

Metrics computed on predictions vs. ground truth:

- Bounding box IoU calculation
- Class prediction accuracy
- Tracking association accuracy

## Logging

Evaluation logs are saved to:

```
{model.logs_dir}/evaluation_{checkpoint_name}.log
```

Logs include:
- Evaluation progress
- Per-sequence metrics
- Overall metrics
- Detailed statistics

## Example Evaluation Output

```
INFO: Loading checkpoint: /path/to/checkpoint.pt
INFO: Evaluating on test set...
INFO: Processing sequence: test_day_0001_td.h5
INFO: Sequence test_day_0001_td.h5:
      mAP@0.5: 0.45
      mAP@0.5:0.95: 0.32
      Precision: 0.67
      Recall: 0.58
INFO: Overall Results:
      mAP@0.5: 0.42
      mAP@0.5:0.95: 0.30
      Precision: 0.65
      Recall: 0.55
```

## Best Practices

### 1. Use Validation Set During Training

Monitor validation metrics during training to avoid overfitting.

### 2. Evaluate on Test Set

Only evaluate on test set after final model selection.

### 3. Per-Class Analysis

Analyze per-class metrics to identify weak classes:

```python
# Check per-class AP
for class_name, ap in per_class_ap.items():
    print(f"{class_name}: {ap:.3f}")
```

### 4. Temporal Analysis

Evaluate temporal consistency:

- Check tracking continuity
- Analyze ID switches
- Monitor track fragmentation

## Troubleshooting

### Common Issues

1. **Low mAP**: Check data quality, verify annotations
2. **High False Positives**: Adjust confidence threshold
3. **Tracking Failures**: Check tracking features, adjust ByteTracker parameters
4. **Memory Issues**: Reduce batch size or sequence length

## Next Steps

- [Performance & Optimizations](./performance)
- [Quick Start Guide](./quick-start)
