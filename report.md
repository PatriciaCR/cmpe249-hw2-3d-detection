# 3D Object Detection Benchmarking (PointPillars & SECOND)

**Patricia Cáceres - CMPE 276 - Fall 2025**

## 1. Goal & Scope
This project evaluates **3D object detection** performance using **two models** (PointPillars and SECOND) across **two datasets** (KITTI and nuScenes-mini). The objectives are to (i) run inference and save all artifacts (frames, point clouds, metadata), (ii) visualize detections locally with Open3D, and (iii) compare models using quantitative metrics and qualitative results.

---
## 2. Setup & Environment
**Compute:** Google Cloud VM (Ubuntu 22.04), NVIDIA L4 GPU, CUDA 11.7 (PyTorch runtime), Python 3.8 (Conda: `mmdet3d-py38`).

**Core Software:** PyTorch 1.13.1+cu117, MMEngine 0.10.7, MMCV 2.0.1, MMDetection3D 1.3.0, Open3D 0.19.0.

**Environment (reproducible commands):**
```bash
conda create -n mmdet3d-py38 python=3.8
conda activate mmdet3d-py38
pip install torch==1.13.1+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmengine==0.10.7 mmcv==2.0.1 mmdet3d==1.3.0
pip install open3d==0.19.0 opencv-python-headless matplotlib
pip install nuscenes-devkit shapely
```
CUDA verification:
```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

---
## 3. Models & Datasets
**Models:**
- **PointPillars** (pillar-based voxelization; strong speed–accuracy tradeoff)
- **SECOND** (sparse 3D convolutions; higher complexity)

**Datasets:**
- **KITTI** (7,481 frames; aligned LiDAR + annotations)
- **nuScenes-mini** (404 frames; used for runtime/qualitative analysis)

---
## 4. Inference, Saving & Visualization
**Inference (modified script):**
```bash
# KITTI
python mmdet3d_inference2.py --dataset kitti --models pointpillars,second \
  --modality lidar --input-path ~/datasets/kitti/training --frame-number -1 \
  --out-dir ~/results_kitti --headless

# nuScenes-mini
python mmdet3d_inference2.py --dataset nuscenes --models pointpillars \
  --modality lidar --input-path ~/datasets/nuscenes --frame-number -1 \
  --out-dir ~/results_nuscenes_pointpillars --headless

python mmdet3d_inference2.py --dataset nuscenes --models second \
  --modality lidar --input-path ~/datasets/nuscenes --frame-number -1 \
  --out-dir ~/results_nuscenes_second --headless
```
**Artifacts saved:** per-frame **.png** (overlays), **.ply** (point clouds + predictions), **.json** (metrics), and a **demo video** stitched from frames.

**Local visualization:** Open3D used to inspect `.ply` files; screenshots captured showing detected objects.

---
## 5. Metrics & Results
To compare models **across datasets**, we report **≥2 evaluation metrics** covering both **accuracy** and **efficiency**. Specifically, we use **IoU, precision, recall** (accuracy-related) and **FPS (latency)** (runtime-related). These metrics enable a clear comparison of what works well, where each model fails, and why.

Frame-level metrics (IoU, precision, recall, FPS) were aggregated to dataset-level averages.

**Table 1 - Average Performance (Accuracy + Runtime Metrics)**

| Model | Dataset | Avg IoU | Precision | Recall | FPS |
||---:|---:|---:|


> *Note:* nuScenes-mini does not provide compatible 3D ground-truth boxes for quantitative accuracy evaluation in this setup; therefore, **IoU/precision/recall are not meaningful**, while **FPS remains a valid comparative metric across datasets**.

| Model | Dataset | Avg IoU | Precision | Recall | FPS |
|------|---------|--------:|----------:|-------:|----:|
| PointPillars | KITTI | 0.516 | 0.419 | 0.672 | 6.79 |
| SECOND | KITTI | 0.209 | 0.027 | 0.042 | 6.21 |
| PointPillars | nuScenes-mini | — | — | — | 15.50 |
| SECOND | nuScenes-mini | — | — | — | 10.39 |

> *Note:* nuScenes-mini lacks compatible 3D ground truth for quantitative accuracy; FPS remains informative.

---
## 6. Qualitative Results
- **KITTI (PointPillars):** consistent car detections with reasonable box alignment.
- **KITTI (SECOND):** frequent misalignment and false positives with the used pretrained checkpoint.
- **nuScenes-mini:** higher FPS due to lower point density; qualitative overlays included.

*(Insert ≥4 labeled screenshots from `results/` here.)*

---
## 7. Comparison & Takeaways
A cross‑dataset, cross‑model comparison reveals clear patterns in **accuracy**, **generalization behavior**, and **computational efficiency**.

### **Model Strengths & Weaknesses (What Works & Why)**
**PointPillars** consistently outperforms SECOND on KITTI across all accuracy metrics-IoU, precision, and recall. Its architecture relies on **pillar-based pseudo‑image representation**, which preserves local geometry while enabling efficient 2D convolutions. This design is well aligned with the spatial statistics of KITTI (sparse automotive LiDAR), enabling robust feature extraction and stable bounding box regression.

In contrast, **SECOND**-a more expressive voxel‑based 3D sparse‑CNN architecture-fails to generalize well with the pretrained checkpoint used here. The misalignment between KITTI’s point cloud distribution and the pretrained model’s expected feature patterns results in **low precision and recall**, indicating both **under‑detection** (missed objects) and **mislocalization** (misaligned boxes). Without dataset‑specific fine‑tuning, SECOND’s higher representational capacity is not effectively utilized.

### **Cross‑Dataset Behavior (Why Models Fail or Succeed Across Domains)**
On nuScenes‑mini, neither model achieves meaningful accuracy because the subset lacks aligned 3D labels for evaluation. However, the dataset still reveals **runtime differences**:
- Both models achieve significantly higher FPS due to nuScenes’ **lower point density** and simpler preprocessing needs.
- PointPillars remains faster than SECOND, consistent with its 2D‑CNN backbone.

These results underscore how **sensor configuration**, **annotation format**, and **point cloud density** strongly influence model behavior. Generalization is not guaranteed even for popular pretrained models.

### **Key Takeaways**
1. **PointPillars achieves the best overall accuracy on KITTI**, demonstrating strong recall (0.67) and moderate precision (0.42), indicating effective geometry encoding and stable detection.
2. **SECOND performs poorly without fine‑tuning**, revealing that voxel‑based architectures are sensitive to dataset shifts and require aligned training distributions.
3. **Runtime differences reflect model complexity and dataset density**-PointPillars is consistently faster, and both models run nearly 2×–3× faster on nuScenes‑mini.
4. **Pretrained models do not universally transfer**; cross‑dataset performance depends heavily on annotation conventions and LiDAR characteristics.
5. **Accuracy metrics and efficiency metrics jointly reveal trade‑offs**: PointPillars dominates in accuracy *and* maintains competitive FPS, making it more deployment‑ready under these conditions.

---
## 8. Limitations & Future Work
- nuScenes-mini prevents quantitative accuracy reporting; full nuScenes would enable mAP/AP.
- SECOND requires **fine-tuning** on KITTI to realize its potential.
- Evaluation uses **BEV IoU** (approximation); full 3D IoU would be stronger.
- Future work: train/fine-tune on a non-KITTI dataset and report learning curves and validation metrics.

---
## 9. Deliverables
- `report.md` (this file)
- `results/` (demo video, ≥4 screenshots, .ply, .json)
- Modified inference code (commented) + README with exact commands and seeds

