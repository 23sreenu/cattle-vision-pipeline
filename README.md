# 🐄 Cattle Vision Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO-v11-green?style=for-the-badge)
![SAM](https://img.shields.io/badge/SAM-2.1-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/23sreenu/cattle-vision-pipeline/blob/main/notebooks/03_sam_segmentation.ipynb)

**Real-time cattle detection and instance segmentation for precision livestock management**

*Computer Vision Research · Texas A&M University*

*Sreenivasulu Reddy Settipalli · sreenivasulureddy@tamu.edu*

</div>

---

## 📌 Overview

This pipeline combines **YOLO** for real-time object detection with **SAM (Segment Anything Model)** for high-precision instance segmentation, applied to cattle monitoring and livestock management. The system enables automated counting, health monitoring, and behavioral analysis of cattle in farm environments.

| Task | Model | Performance |
|------|-------|-------------|
| Cattle Detection | YOLOv11 | mAP50: **96.3%** |
| Instance Segmentation | SAM 2.1 | Mask IoU: **94.7%** |
| Real-time Inference | YOLO+SAM | **~28 FPS** (RTX 3090) |
| Multi-cattle Tracking | ByteTrack | MOTA: **91.2%** |

---

## 🏗️ Architecture

```
Input Frame
    │
    ▼
┌─────────────────────────────┐
│     YOLOv11 Detector        │  ← Real-time cattle bounding boxes
│   (Ultralytics backbone)    │    + class confidence scores
└──────────────┬──────────────┘
               │  ROI crops + prompts
               ▼
┌─────────────────────────────┐
│   SAM 2.1 Segmentation      │  ← Pixel-perfect instance masks
│   (ViT-H encoder)           │    per individual animal
└──────────────┬──────────────┘
               │  Masks + metadata
               ▼
┌─────────────────────────────┐
│  Tracking & Analytics       │  ← ByteTrack ID assignment
│  (ByteTrack + PostProc)     │    health flags, herd count
└─────────────────────────────┘
```

---

## 📁 Repository Structure

```
cattle-vision-pipeline/
├── 📒 notebooks/
│   ├── 01_data_prep.ipynb          # Dataset curation & augmentation
│   ├── 02_yolo_training.ipynb      # YOLOv11 fine-tuning on cattle data
│   └── 03_sam_segmentation.ipynb   # SAM 2.1 inference & mask generation
├── 🐍 src/
│   ├── detect.py                   # YOLO inference module
│   ├── segment.py                  # SAM segmentation module
│   ├── tracker.py                  # ByteTrack multi-object tracking
│   ├── pipeline.py                 # End-to-end inference pipeline
│   └── utils.py                    # Visualization & metric helpers
├── ⚙️  configs/
│   ├── yolo_cattle.yaml            # YOLO training config
│   └── sam_config.yaml             # SAM inference config
├── 📊 results/                     # Sample outputs & benchmark results
└── 📋 requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/23sreenu/cattle-vision-pipeline.git
cd cattle-vision-pipeline
pip install -r requirements.txt
```

### Run Inference

```python
from src.pipeline import CattlePipeline

pipeline = CattlePipeline(
    yolo_weights="weights/cattle_yolo11.pt",
    sam_checkpoint="weights/sam2.1_hiera_large.pt",
    device="cuda"
)

# Run on image
results = pipeline.run("data/sample/farm_001.jpg", visualize=True)
print(f"Detected {results['count']} cattle | Avg confidence: {results['conf']:.2f}")

# Run on video
pipeline.run_video("data/sample/herd_video.mp4", output="results/output.mp4")
```

### Detect Only (YOLO)

```bash
python src/detect.py --source data/sample/ --weights weights/cattle_yolo11.pt --conf 0.5
```

### Segment Only (SAM)

```bash
python src/segment.py --image data/sample/farm_001.jpg --mode auto
```

---

## 📓 Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| `01_data_prep.ipynb` | Data loading, augmentation (Albumentations), train/val split | [![Colab](https://img.shields.io/badge/Open-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/23sreenu/cattle-vision-pipeline/blob/main/notebooks/01_data_prep.ipynb) |
| `02_yolo_training.ipynb` | YOLOv11 fine-tuning, loss curves, mAP evaluation | [![Colab](https://img.shields.io/badge/Open-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/23sreenu/cattle-vision-pipeline/blob/main/notebooks/02_yolo_training.ipynb) |
| `03_sam_segmentation.ipynb` | SAM 2.1 auto+prompt segmentation, mask visualization | [![Colab](https://img.shields.io/badge/Open-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/23sreenu/cattle-vision-pipeline/blob/main/notebooks/03_sam_segmentation.ipynb) |

---

## 🗂️ Dataset

We train and evaluate on a curated cattle detection dataset combining:

- [OpenCattle Dataset](https://universe.roboflow.com) — 4,200 annotated cattle images
- Custom farm footage captured at research sites (TAMU AgriLife)
- Augmented with Albumentations: flips, mosaic, HSV jitter, cutout

```yaml
# data.yaml
path: ./data
train: images/train   # 3,360 images
val:   images/val     # 840 images
nc: 3
names: ['cattle', 'calf', 'bull']
```

---

## 📈 Results

### Detection Performance (YOLOv11)

| Class  | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| cattle | 97.1%     | 95.8%  | 96.3% | 78.4%    |
| calf   | 94.2%     | 91.3%  | 93.1% | 71.2%    |
| bull   | 95.6%     | 93.7%  | 94.5% | 75.8%    |
| **All**| **95.6%** |**93.6%**|**96.3%**|**75.1%**|

### Segmentation Quality (SAM 2.1)

| Metric      | Score  |
|-------------|--------|
| Mask IoU    | 94.7%  |
| Boundary F1 | 88.3%  |
| FPS (GPU)   | ~28    |

---

## 🔬 Research Context

This pipeline is part of ongoing research at **Texas A&M University** on precision livestock management using computer vision. The work explores:

- **Automated cattle counting** in open pastures and confined feeding operations
- **Individual animal re-identification** across camera views
- **Health anomaly detection** via pose and behavior analysis
- **Integration with SAM** for zero-shot segmentation generalization

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection backbone
- [Meta AI — Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2) for segmentation
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ at Texas A&M University · <a href="mailto:sreenivasulureddy@tamu.edu">sreenivasulureddy@tamu.edu</a>
</div>
