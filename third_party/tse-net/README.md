# TSE-Net: Semi-supervised Monocular Height Estimation from Single Remote Sensing Images

This repository provides the implementation for [paper](). We introduce a semi-supervised learning pipeline for monocular height estimation that integrates regression and classification tasks to enhance pseudo-label filtering and improve learning from unlabeled data.

---

## 🛠️ Installation

> ⚠️ **Note:** The provided environment file is for **reference only**. Some packages may require manual installation or configuration.

### Recommended Environment
- `pytorch` 1.7.1  
- `scikit-image` 0.21.0  
- `wandb` (for experiment tracking)

---

## 🚀 Usage

### ⚙️ Configuration File

A configuration file is needed to launch training, see `configs/*.yaml` as references. For the model and training configuration, most hyperparameters are set as defaults in `tsenet.py`, but you may override them here in the configs if desired.

For the data configurations, you must define:
   - `data_dir`: path to your dataset  
   - `data_train`, `data_val`, and `data_test`: list of text files representing the data split. 
The parameters for DFC23 dataset are provided as reference.

---

### 📂 Data Preparation

Organize your dataset in the following structure under `data_dir`:
```
📂 data_dir
    📂 image # Opitcal satellite images
    📂 mask # land cover masks (not as network input, only for computing building metrics)
    📂 ndsm (Ground truth normalized DSMs)
```

Each scene should have the same filename base, e.g., `scene_001`, with different suffixes:
- `_IMG.tif` – optical image  
- `_BLG.tif` – land cover masks (optional)  
- `_AGL.tif` – nDSM height map


**Example:**
```
scene_001_IMG.tif
scene_001_BLG.tif
scene_001_AGL.tif
```

Define your data splits in test files defined in `data_train`, `data_val`, and `data_test`.
Each file lists scene bases (without extensions), e.g.:
```
scene_001
scene_002
...
scene_xxx
```
For train, validation and test, data entries from all split files will be concatenated to build the dataloader. 

---

### 🎯 Training
Before training, run the following command to compute the HB cut points for each dataset.
```bash
python compute_hbc.py
```
Please edit the data and split folders before running.

To start training with the provided example configuraton, simply run
```bash
python train.py --exp_config /path/to/saved/config --restore
```
After training, there will be several checkpoint files under the checkpoint directory, `checkpoint_last.pth.tar` for the last epoch, `checkpoint_best_rmse.pth.tar` for the epoch with best validation RMSE, and so on.

---

### 📊 Evaluation
Evaluate a trained model with:
```bash
python test.py --config /path/to/archived/config/under/checkpoint/directory test_checkpoint_file checkpoint_best_rmse.pth.tar
```
Replace checkpoint_best_rmse.pth.tar with any other saved checkpoint as needed. The results with be save as `result_best_rmse.pth.tar`.
