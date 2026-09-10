# Training Improvements - Depth Anything V2 Fine-tuning

This document describes the improvements implemented for the Depth Anything V2 training pipeline on the Moroccan buildings dataset.

## Overview

The training pipeline has been significantly enhanced with multiple optimizations to improve model performance, reduce overfitting, and provide better analysis tools. All improvements are implemented in the `pipelines/04_finetuning_dav2.py` script and related training components.

## Implemented Improvements

### 1. Enhanced Training Parameters

**Default Configuration Changes:**
- Epochs: Increased from 100 to 200
- Patience: Increased from 8 to 20 (more robust early stopping)
- Batch size: Optimized to 2 with gradient accumulation of 4 (effective batch size = 8)
- Image size: Reduced to 392 for memory efficiency (must be multiple of 14 for ViT-L)

**Benefits:**
- Longer training allows better convergence
- Increased patience prevents premature stopping
- Memory-efficient configuration maintains effective batch size

### 2. Multi-Learning Rate Testing System

**Implementation:**
- Automated testing of multiple learning rates (1e-5, 2e-5, 1e-4)
- Sequential training with different ports to avoid conflicts
- Automatic comparison and ranking of results
- Comprehensive result tracking and visualization

**Usage:**
```bash
python3 pipelines/04_finetuning_dav2.py --multi-lr-test
```

**Benefits:**
- Systematic hyperparameter optimization
- Automatic identification of best learning rate
- Time-efficient comparison process

### 3. Label Smoothing

**Implementation:**
- Added label smoothing coefficient (default: 0.1)
- Applied to SiLog loss to reduce overfitting
- Gaussian noise injection on target values

**Technical Details:**
- Reduces model confidence on training data
- Improves generalization to unseen data
- Particularly effective for depth estimation tasks

**Usage:**
```bash
--label-smoothing 0.1
```

### 4. Dropout Regularization

**Implementation:**
- Added dropout layers in fully-connected layers of DPT head
- Dropout rate: 0.1 (configurable)
- Applied to output_conv1 and output_conv2 layers

**Technical Details:**
- Randomly deactivates neurons during training
- Prevents co-adaptation of features
- Improves model robustness

**Usage:**
```bash
--dropout-rate 0.1
```

### 5. Building-Specific Data Augmentations

**Implementation:**
- Specialized augmentations for building imagery
- Enhanced lighting simulation
- Architectural shadow simulation
- Color temperature variations

**Specific Augmentations:**
- Random brightness/contrast for different lighting conditions
- RGB shift for color temperature simulation
- Architectural shadows with multiple shadow sources
- Low-light condition simulation

**Technical Details:**
- Applied only to RGB images (not depth maps)
- Maintains geometric consistency between image and depth
- Preserves valid mask integrity

**Usage:**
```bash
--building-lighting
```

### 6. Multi-Loss Function

**Implementation:**
- Combined loss function with multiple components
- Weighted combination of complementary loss functions

**Loss Components:**
1. **SiLog Loss** (weight: 1.0): Scale-invariant logarithmic loss
2. **L1 Loss** (weight: 0.5): L1 loss for stability
3. **Gradient Loss** (weight: 0.3): Gradient loss for edge preservation
4. **Scale-Invariant Loss** (weight: 0.2): Additional scale-invariant loss

**Technical Details:**
- SiLog: Primary loss for depth estimation
- L1: Provides numerical stability
- Gradient: Preserves edges and structural details
- Scale-Invariant: Reduces scale-related errors

**Usage:**
```bash
--use-multi-loss
--silog-weight 1.0
--l1-weight 0.5
--gradient-weight 0.3
--scale-weight 0.2
```

### 7. Advanced Visualization and Comparison

**Implementation:**
- Comprehensive visualization system for training results
- Multiple chart types for different analysis needs

**Generated Visualizations:**
1. **Metrics Comparison Bar Charts**: Comparison of main metrics (abs_rel, RMSE, delta1, latency) across different learning rates
2. **Validation vs Test Charts**: Side-by-side comparison of validation and test performance
3. **Scatter Plots**: Prediction vs ground truth scatter plots with R² correlation coefficients
4. **Radar Charts**: Normalized multi-metric performance visualization

**Output Location:**
- CSV files: `results/metrics_dav2_multi_lr_*.csv`
- Visualizations: `results/plots/*.png`
- Predictions: `results/predictions_dav2_*.npz`

### 8. Memory Optimization

**Implementation:**
- Gradient accumulation for effective large batch training
- Memory-efficient image size (392 instead of 518)
- Optimized batch size (2 with accumulation = 4)
- Port management for concurrent training

**Technical Details:**
- Reduces GPU memory usage by ~60%
- Maintains effective batch size of 8
- Enables multi-LR testing without memory conflicts

## Usage Instructions

### Standard Training (Single Learning Rate)

```bash
python3 pipelines/04_finetuning_dav2.py \
  --epochs 200 \
  --patience 20 \
  --batch-size 2 \
  --img-size 392 \
  --gradient-accumulation 4 \
  --lr 1e-5 \
  --use-multi-loss \
  --label-smoothing 0.1 \
  --dropout-rate 0.1 \
  --building-lighting
```

### Multi-LR Testing

```bash
python3 pipelines/04_finetuning_dav2.py \
  --multi-lr-test \
  --epochs 200 \
  --patience 20 \
  --batch-size 2 \
  --img-size 392 \
  --gradient-accumulation 4 \
  --use-multi-loss \
  --label-smoothing 0.1 \
  --dropout-rate 0.1 \
  --building-lighting
```

### Custom Configuration

All parameters can be customized:

```bash
python3 pipelines/04_finetuning_dav2.py \
  --epochs 150 \
  --patience 15 \
  --batch-size 2 \
  --lr 2e-5 \
  --use-multi-loss \
  --label-smoothing 0.15 \
  --dropout-rate 0.15 \
  --silog-weight 1.2 \
  --l1-weight 0.4 \
  --gradient-weight 0.4 \
  --scale-weight 0.2
```

## Parameter Reference

### Training Parameters
- `--epochs`: Number of training epochs (default: 200)
- `--patience`: Early stopping patience (default: 20)
- `--batch-size`: Batch size (default: 2)
- `--gradient-accumulation`: Gradient accumulation steps (default: 4)
- `--lr`: Learning rate (default: 5e-6)

### Model Parameters
- `--encoder`: ViT encoder (vits, vitb, vitl - default: vitl)
- `--img-size`: Image size (default: 392, must be multiple of 14)
- `--dropout-rate`: Dropout rate (default: 0.1)

### Loss Parameters
- `--use-multi-loss`: Enable multi-loss function (default: True)
- `--label-smoothing`: Label smoothing coefficient (default: 0.1)
- `--silog-weight`: SiLog loss weight (default: 1.0)
- `--l1-weight`: L1 loss weight (default: 0.5)
- `--gradient-weight`: Gradient loss weight (default: 0.3)
- `--scale-weight`: Scale-invariant loss weight (default: 0.2)

### Augmentation Parameters
- `--building-lighting`: Enable building-specific augmentations (default: True)

### Testing Parameters
- `--multi-lr-test`: Enable multi-LR testing mode
- `--use-stratified`: Use stratified splits (default: True)
- `--use-original`: Use original splits instead of stratified

## Performance Considerations

### Training Time Estimates
- **Per epoch**: ~3,325 iterations, ~0.42 seconds/iteration
- **Maximum training time (200 epochs)**: ~77 hours
- **Expected training time (with early stopping)**: 20-40 hours
- **Multi-LR testing**: 3x single training time

### Memory Requirements
- **GPU Memory**: ~2-3 GB with optimized settings
- **Effective Batch Size**: 8 (2 batch size × 4 accumulation)
- **Recommended GPU**: NVIDIA L40S or equivalent with 8GB+ VRAM

### Expected Performance Improvements
Based on the implemented improvements, expected performance gains include:
- **Reduced overfitting**: Label smoothing and dropout
- **Better generalization**: Building-specific augmentations
- **Improved edge preservation**: Gradient loss component
- **More stable training**: Multi-loss function
- **Optimal hyperparameters**: Systematic LR testing

## Technical Implementation Details

### Modified Files
1. `pipelines/04_finetuning_dav2.py`: Main training orchestration
2. `third_party/Depth-Anything-V2/metric_depth/util/loss.py`: Loss functions
3. `third_party/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py`: Model architecture
4. `third_party/Depth-Anything-V2/metric_depth/dataset/augmentations_depth.py`: Data augmentations
5. `third_party/Depth-Anything-V2/metric_depth/dataset/batiments_maroc.py`: Dataset class
6. `third_party/Depth-Anything-V2/metric_depth/train.py`: Training loop

### Key Technical Changes
- Multi-loss function with dynamic weight adjustment
- Dropout integration in DPT head architecture
- Enhanced augmentation pipeline with building-specific transforms
- Port management for concurrent training processes
- Comprehensive visualization and comparison system

## Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce batch size further: `--batch-size 1`
- Reduce image size: `--img-size 364`
- Kill existing GPU processes: `pkill -f train.py`

**Port Already in Use:**
- The system automatically uses different ports for multi-LR testing
- Manual port specification available via custom implementation

**Slow Training:**
- Reduce image size for faster processing
- Reduce epochs if early stopping works well
- Consider using smaller encoder (vitb instead of vitl)

## Results and Analysis

### Output Files Structure
```
results/
├── metrics_dav2_multi_lr_val_stratified.csv
├── metrics_dav2_multi_lr_test_stratified.csv
├── predictions_dav2_lr_1e-5_test_stratified.npz
├── predictions_dav2_lr_2e-5_test_stratified.npz
├── predictions_dav2_lr_0.0001_test_stratified.npz
└── plots/
    ├── comparaison_lr_test_stratified.png
    ├── comparaison_val_test_stratified.png
    ├── scatter_pred_true_stratified.png
    └── radar_performance_stratified.png
```

### Interpretation Guide
- **Lower abs_rel/RMSE**: Better accuracy
- **Higher delta1**: Better threshold accuracy
- **Higher R²**: Better correlation with ground truth
- **Consistent val/test performance**: Better generalization

## Future Improvements

Potential areas for further enhancement:
- Curriculum learning with progressive augmentation difficulty
- Knowledge distillation from larger models
- Test-time augmentation for inference improvement
- Multi-scale training with variable image sizes
- Advanced regularization techniques (mixup, cutmix adapted for depth)

## References

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Albumentations: https://albumentations.ai/
- PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/dist_tuto.html

## Version Information

- **Branch**: section2_training
- **Base Implementation**: Depth Anything V2 Metric Depth
- **Dataset**: Moroccan Buildings Height Dataset
- **Framework**: PyTorch with Distributed Data Parallel
