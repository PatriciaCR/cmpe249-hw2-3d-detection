````markdown
# 3D Object Detection Benchmarking  
## PointPillars & SECOND on KITTI + nuScenes-mini  
**Patricia Cáceres — CMPE 276 — Fall 2025**

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.13.1%2Bcu117-red.svg)]()
[![MMDetection3D](https://img.shields.io/badge/mmdet3d-1.3.0-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 📌 Overview

This repository benchmarks **PointPillars** and **SECOND** on the **KITTI** and **nuScenes-mini** datasets using an extended MMDetection3D inference pipeline. The system supports:

- Height-colored LiDAR visualization  
- 3D Open3D rendering (headless or GUI)  
- KITTI 3D→2D projection overlays  
- IoU, precision, recall, FPS computation  
- Automatic `.mp4` video demo generation  
- Unified inference across KITTI and nuScenes-mini LiDAR datasets  

All experiments were executed on a Google Cloud VM with an NVIDIA L4 GPU and full CUDA acceleration.

---

# 🚀 1. Compute Environment

### **Google Cloud VM**
| Component | Specification |
|----------|--------------|
| Machine Type | g2-standard-8 (8 vCPU, 32GB RAM) |
| GPU | NVIDIA L4 (Compute Capability 8.9) |
| OS | Ubuntu 22.04 LTS |
| Python | 3.8.20 |
| CUDA Runtime (PyTorch) | 11.7 |
| NVIDIA Driver | 535.274.02 |

> **Note:** `nvcc` was not installed system-wide; however, PyTorch correctly detected and used the CUDA runtime provided by the NVIDIA driver.

---

# 🎯 2. Installation (Reproducible)

### Create Conda Environment
```bash
conda create -n mmdet3d-py38 python=3.8
conda activate mmdet3d-py38
````

### Install PyTorch (CUDA 11.7)

```bash
pip install torch==1.13.1+cu117 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install Core OpenMMLab Stack

```bash
pip install mmengine==0.10.7
pip install mmcv==2.0.1
pip install mmdet3d==1.3.0
```

### Visualization & Utility Dependencies

```bash
pip install open3d==0.19.0
pip install opencv-python-headless
pip install matplotlib
pip install shapely
pip install nuscenes-devkit
```
nuscenes-devkit is required for indexing nuScenes-mini.
KITTI parsing is handled directly by MMDetection3D utilities.

### (Optional) Clone MMDetection3D

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

---

# 🔍 3. Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import mmengine; print(mmengine.__version__)"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmdet3d; print(mmdet3d.__version__)"
nvidia-smi
```

Expected:

* mmengine **0.10.7**
* mmcv **2.0.1**
* mmdet3d **1.3.0**
* CUDA **11.7**

---

# 📁 4. Dataset Setup

### KITTI Directory Structure

```
kitti/
 ├── training/
 │    ├── velodyne/
 │    ├── image_2/
 │    ├── calib/
 │    ├── label_2/
```

### nuScenes-mini Directory Structure

```
nuscenes/
 ├── samples/
 ├── sweeps/
 ├── v1.0-mini/
```

---

# 🧠 5. Running Inference

### KITTI — Run PointPillars & SECOND

```bash
python mmdet3d_inference2.py \
    --dataset kitti \
    --models pointpillars,second \
    --modality lidar \
    --input-path ~/datasets/kitti/training \
    --frame-number -1 \
    --out-dir ~/results_kitti \
    --headless
```

---

### nuScenes-mini — PointPillars

```bash
python mmdet3d_inference2.py \
    --dataset nuscenes \
    --models pointpillars \
    --modality lidar \
    --input-path ~/datasets/nuscenes \
    --frame-number -1 \
    --out-dir ~/results_nuscenes_pointpillars \
    --headless
```

### nuScenes-mini — SECOND

```bash
python mmdet3d_inference2.py \
    --dataset nuscenes \
    --models second \
    --modality lidar \
    --input-path ~/datasets/nuscenes \
    --frame-number -1 \
    --out-dir ~/results_nuscenes_second \
    --headless
```

---

# 📊 6. Metrics Generation

### Per-frame Metrics (Generated During Inference)

```
<frame_id>_metrics.json
```
Each file contains:
 * Average BEV IoU (GT → best prediction match)
 * Precision
 * Recall
 * FPS / inference latency
 * Number of predicted & GT objects

KITTI: full IoU / precision / recall
nuScenes-mini: timing and prediction counts only (no aligned GT)

### Aggregate metrics across frames

A separate aggregation script computes dataset-level statistics by averaging per-frame metrics:
```bash
python compute_metrics.py \
    --metrics-dir ~/results_kitti_second \
    --out kitti_second_aggregate.json
```

Metrics include:

* Mean IoU
* Mean precision / recall
* Mean FPS
* Total frames processed

This separation ensures reproducibility and prevents loss of metrics if inference is interrupted.
---

# 🎞 7. Video Demo Creation

```bash
ffmpeg -framerate 10 -pattern_type glob \
    -i "*_2d_vis.png" \
    -c:v libx264 -pix_fmt yuv420p demo_video.mp4
```

---

# 🔐 8. Random Seed (Reproducible Outcomes)

```python
import random, numpy as np, torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

# 🐞 9. Troubleshooting

* nuScenes-mini does not provide frame-aligned GT for IoU evaluation
* BEV IoU approximation ignores height overlap
* No official mAP evaluation (focus is inference benchmarking)
* Headless Open3D visualization required on cloud VMs
---

# 📘 10. License

MIT License © Patricia Cáceres — 2025
