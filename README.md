# CRTS2 Building Height Estimation Benchmark

A comprehensive benchmark framework for evaluating building height estimation models on Moroccan building datasets. This project compares state-of-the-art monocular depth estimation models including DepthPro, Depth Anything V2, HTC-DC Net, and TSE-Net.

## 📋 Overview

This benchmark provides a complete pipeline for:
- **Data preparation**: Processing geospatial data (GeoTIFF orthophotos, Shapefile building footprints) into training patches
- **Model evaluation**: Zero-shot and fine-tuned evaluation of multiple depth estimation models
- **Metrics computation**: Comprehensive evaluation metrics aligned with academic literature (RMSE, RMSE-B, NMAD, etc.)
- **Statistical analysis**: Wilcoxon signed-rank tests for model comparison significance
- **Spatial validation**: Stratified train/val/test splits with spatial constraints to prevent data leakage

## 🏗️ Project Structure

```
crts2_bench_haut_new20/
├── core/                      # Core utilities
│   ├── data_utils.py         # Data loading and split management
│   ├── geo_io.py             # Geospatial I/O operations
│   └── metrics.py            # Evaluation metrics and statistical tests
├── pipelines/                 # Processing pipeline scripts
│   ├── 00_exploratory_data_analysis.py
│   ├── 01_audit_donnees.py
│   ├── 02_preparation_data.py
│   ├── 03_benchmark_zero_shot.py
│   ├── 04_finetuning_dav2.py
│   ├── 05_train_htc_dc_net.py
│   ├── 06_train_tse_net.py
│   ├── 07_evaluation_finale.py
│   └── 08_lidar_calibration.py
├── configs/                   # Configuration files
│   └── batiments_maroc.yaml
├── requirements/              # Dependency files for each model
│   ├── depth_anything_v2.txt
│   ├── depthpro.txt
│   ├── htc_dc_net.txt
│   └── tse_net.txt
├── third_party/              # Third-party model repositories
│   ├── Depth-Anything-V2/
│   ├── HTC-DC-Net/
│   ├── ml-depth-pro/
│   └── tse-net/
├── data/                      # Data directory (user-provided)
│   ├── zone1/, zone2/, zone3/
│   ├── patches/
│   └── splits/
├── results/                   # Results and outputs
├── checkpoints/              # Model checkpoints
└── setup.sh                  # Initial setup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

### Installation

1. **Clone and setup third-party repositories:**
```bash
bash setup.sh
```

This will:
- Clone all third-party model repositories
- Create virtual environments for each model
- Install dependencies
- Create checkpoint directories

2. **Manual setup steps:**
- Download Depth Anything V2 weights (see `pipelines/download_dav2_weights.py`)
- Train HTC-DC Net and TSE-Net on your data (no public weights available)
- Place your data in `data/zone1/`, `data/zone2/`, `data/zone3/`

### Data Preparation

1. **Exploratory Data Analysis:**
```bash
python pipelines/00_exploratory_data_analysis.py
```

2. **Data Audit:**
```bash
python pipelines/01_audit_donnees.py
```

3. **Data Preparation:**
```bash
python pipelines/02_preparation_data.py
```

4. **Create Stratified Split (optional):**
```bash
python create_stratified_split.py
```

This creates spatially-constrained train/val/test splits to prevent data leakage.

## 📊 Model Evaluation Pipeline

### 1. Zero-Shot Benchmark

Evaluate DepthPro and Depth Anything V2 without training:
```bash
python pipelines/03_benchmark_zero_shot.py
```

### 2. Fine-Tuning Depth Anything V2

Fine-tune DAV2 on your dataset:
```bash
python pipelines/04_finetuning_dav2.py
```

### 3. Train HTC-DC Net

Train HTC-DC Net from scratch:
```bash
python pipelines/05_train_htc_dc_net.py
```

Test GPU memory requirements:
```bash
python test_htc_batch_size.py
```

### 4. Train TSE-Net

Train TSE-Net on your dataset:
```bash
python pipelines/06_train_tse_net.py
```

### 5. Final Evaluation

Generate comprehensive comparison report:
```bash
python pipelines/07_evaluation_finale.py
```

This produces:
- Final comparison table with all metrics
- Wilcoxon significance matrix
- Per-building error analysis

## 📈 Evaluation Metrics

The benchmark uses metrics aligned with academic literature:

### Pixel-Level Metrics (Technical Diagnostics)
- **RMSE**: Root Mean Square Error (all pixels)
- **RMSE-M**: RMSE on building pixels only
- **RMSE-NM**: RMSE on non-building pixels only

### Building-Level Metrics (Decision Metrics)
- **MAE**: Mean Absolute Error
- **RMSE-B**: RMSE per building (median aggregation)
- **NMAD**: Normalized Median Absolute Deviation (robust to outliers)
- **Bias**: Systematic error (mean prediction - ground truth)
- **Relative Error**: Mean absolute relative error
- **Correlation**: Pearson correlation coefficient
- **Coverage**: Percentage of buildings successfully evaluated

### Statistical Significance
- **Wilcoxon signed-rank test**: Paired test between models
- **p-value < 0.05**: Statistically significant difference

## 🔧 Configuration

### HTC-DC Net Configuration

Edit `configs/batiments_maroc.yaml`:

```yaml
model: htcdc
backbone: efficientnetb0
project: BatimentsMaroc

data_dir: ../../../data/htc_dc_net_format
batch_size: 8
image_size: 256
use_mask: True

max_epochs: 200
lr: 0.0001
chamfer_weight: 0.01
```

### Data Format

Expected data structure:
```
data/
├── zone1/
│   ├── orthophoto.tif
│   └── buildings.shp
├── zone2/
│   ├── orthophoto.tif
│   └── buildings.shp
└── zone3/
    ├── orthophoto.tif
    └── buildings.shp
```

Shapefiles must contain a `HAUTEUR` field with building heights in meters.

## 🧪 Testing and Validation

### Verify Data Loading
```bash
python verif.py
```

This generates visual verification of patches (orthophoto, height map, building mask).

### Test GPU Memory
```bash
python test_htc_batch_size.py
```

Tests different batch sizes for HTC-DC Net to find optimal configuration.

## 📁 Output Files

Results are saved in the `results/` directory:
- `predictions_*.npz`: Raw predictions per model
- `tableau_final_comparatif.csv`: Final comparison table
- `matrice_significativite.csv`: Statistical significance matrix
- `stratified_split/`: Split validation reports and visualizations

## 🔬 Models Compared

### 1. DepthPro (Apple)
- **Type**: Zero-shot monocular depth estimation
- **Architecture**: Transformer-based
- **Weights**: Publicly available
- **Paper**: "DepthPro: Sharp Monocular Depth Estimation from Boosting Any Foundation Model"

### 2. Depth Anything V2
- **Type**: Zero-shot / Fine-tunable monocular depth estimation
- **Architecture**: Vision Transformer (ViT-L)
- **Weights**: Publicly available
- **Paper**: "Depth Anything V2"

### 3. HTC-DC Net
- **Type**: Supervised building height estimation
- **Architecture**: EfficientNet backbone with height classification
- **Weights**: Requires training on your data
- **Paper**: "HTC-DC Net: Hierarchical Transformer for Dense Depth Completion"

### 4. TSE-Net
- **Type**: Supervised building height estimation
- **Architecture**: Transformer-based with spatial encoding
- **Weights**: Requires training on your data
- **Paper**: "TSE-Net: Transformer-based Spatial Encoding for Building Height Estimation"

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config files
   - Use `test_htc_batch_size.py` to find optimal batch size

2. **Missing Checkpoints**
   - Download DAV2 weights manually
   - Train HTC-DC Net and TSE-Net first

3. **Data Loading Errors**
   - Verify data structure matches expected format
   - Check Shapefile has `HAUTEUR` field
   - Run `verif.py` to visualize data

4. **Spatial Split Issues**
   - Ensure geographic coordinates are available
   - If not, the script falls back to zone-based clustering

## 📝 Citation

If you use this benchmark in your research, please cite the respective papers of the models evaluated:

```bibtex
@article{depthpro,
  title={DepthPro: Sharp Monocular Depth Estimation from Boosting Any Foundation Model},
  author={...},
  journal={...},
  year={2024}
}

@article{depth_anything_v2,
  title={Depth Anything V2},
  author={...},
  journal={...},
  year={2024}
}

@article{htc_dc_net,
  title={HTC-DC Net: Hierarchical Transformer for Dense Depth Completion},
  author={...},
  journal={arXiv:2309.16486},
  year={2023}
}

@article{tse_net,
  title={TSE-Net: Transformer-based Spatial Encoding for Building Height Estimation},
  author={...},
  journal={arXiv:2511.13552},
  year={2025}
}
```

## 📄 License

This project uses third-party code with their respective licenses:
- DepthPro: Apple License
- Depth Anything V2: Apache 2.0
- HTC-DC Net: See repository
- TSE-Net: See repository

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- All tests pass
- Documentation is updated
- Changes are backwards compatible when possible

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This benchmark is designed for research purposes. Ensure you have proper rights to use the building data and comply with all applicable regulations.
