# 🚧 Worker & Construction Equipment Detection System

## 📌 Overview
This project presents a deep learning-based system for detecting and analyzing interactions between **workers and heavy construction equipment** using YOLOv11.

The system is designed as a foundation for advancing from **reactive detection** to **predictive hazard forecasting**, enabling early safety warnings in construction environments.

---

## 🎯 Objectives
- Detect key construction entities in real-time:
  - Worker
  - Excavator
  - Bulldozer
  - Dump Truck
  - Concrete Mixer
- Enable robust detection under:
  - Occlusion
  - Complex backgrounds
  - Varying object scales
- Prepare the pipeline for **future trajectory prediction and intent modeling (PIHNet)**

---

## 🧠 Methodology

### 🔹 Detection Model
- **Model:** YOLOv11 (Ultralytics)
- **Task:** Object Detection
- **Framework:** PyTorch

### 🔹 Training Strategy
- Custom dataset (annotated using CVAT / Roboflow)
- Data augmentation applied
- High-resolution training for small object detection

---

## 📊 Results

| Metric            | Score |
|------------------|------|
| mAP@0.5          | 0.88 |
| mAP@0.5:0.95     | 0.70 |
| Precision        | 0.88 |
| Recall           | 0.86 |

> *Note: Results may vary depending on dataset and training configuration.*

---

## 📂 Project Structure
├── train_model.py # Training script
├── data.yaml # Dataset configuration
├── README.md # Project documentation
├── .gitignore # Ignore large files
└── docs/
└── sample_output.jpg


## Visual Representation
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/11e45d57-97da-4d5b-b31a-2c189798649b" />












