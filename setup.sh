#!/bin/bash
set -e

echo "=== Clonage des dépôts tiers ==="
git clone https://github.com/apple/ml-depth-pro.git third_party/ml-depth-pro
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
git clone https://github.com/zhu-xlab/HTC-DC-Net.git third_party/HTC-DC-Net
git clone https://github.com/zhu-xlab/tse-net.git third_party/tse-net

echo "=== Création des environnements virtuels ==="
python -m venv --system-site-packages third_party/ml-depth-pro/venv
python -m venv --system-site-packages third_party/Depth-Anything-V2/venv
python -m venv --system-site-packages third_party/HTC-DC-Net/venv
python -m venv --system-site-packages third_party/tse-net/venv

echo "=== Installation DepthPro ==="
source third_party/ml-depth-pro/venv/bin/activate
pip install --upgrade pip
pip install -e third_party/ml-depth-pro
deactivate

echo "=== Installation Depth Anything V2 ==="
source third_party/Depth-Anything-V2/venv/bin/activate
pip install --upgrade pip
pip install -r requirements/depth_anything_v2.txt
deactivate

echo "=== Installation HTC-DC Net ==="
source third_party/HTC-DC-Net/venv/bin/activate
pip install --upgrade pip
pip install -r requirements/htc_dc_net.txt
deactivate

echo "=== Installation TSE-Net ==="
source third_party/tse-net/venv/bin/activate
pip install --upgrade pip
pip install -r requirements/tse_net.txt
deactivate


mkdir -p checkpoints/depthpro checkpoints/depth_anything_v2 checkpoints/htc_dc_net checkpoints/tse_net

echo ""
echo "=== Setup terminé ==="
echo "Reste à faire manuellement :"
echo "  - Poids DAV2       : voir pipelines/download_dav2_weights.py"
echo "  - HTC-DC Net / TSE-Net : pas de poids publics, entraînement requis sur vos données"
echo "  - Déposer vos données dans data/zone1/ data/zone2/ data/zone3/"