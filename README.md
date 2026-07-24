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

## Visual Representation
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/11e45d57-97da-4d5b-b31a-2c189798649b" />

## 📊 Results

| Metric            | Score |
|------------------|------|
| mAP@0.5          | 0.88 |
| mAP@0.5:0.95     | 0.70 |
| Precision        | 0.88 |
| Recall           | 0.86 |

> *Note: Results may vary depending on dataset and training configuration.*

## BoxF1_curve

<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/3fa3111f-5d5e-4fe1-81d8-466509f992bb" />

## BoxP_curve

<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/83be6cf9-662e-46e1-8d50-1289dd296c34" />

## BoxPR_curve

<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/2b827cac-64a0-4c63-93e7-ab8961e48061" />

## BoxR_curve

<img width="2250" height="1500" alt="BoxR_curve" src="https://github.com/user-attachments/assets/4fe58595-14d3-4293-9822-13981267ef06" />

## Confusion_matrix

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/4574e8a9-7f11-48c6-8a75-437ff0664d21" />

## Confusion_matrix_normalized

<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/24698c1c-b176-423b-a168-3df4259f1b53" />

## Labels

<img width="1600" height="1600" alt="labels" src="https://github.com/user-attachments/assets/7b25525a-6e42-4f44-8e97-dc328876f107" />

## Results

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/0451f7f4-f14a-4b60-94cf-daec3b2b8edf" />


## Trains
<img width="1920" height="1920" alt="train_batch1" src="https://github.com/user-attachments/assets/521e9c10-f2b0-4a65-a078-6218e2d572e8" />
 
<img width="1920" height="1920" alt="train_batch503120" src="https://github.com/user-attachments/assets/80aef6a1-74c2-4d81-8fee-cd257271646a" />

<img width="1920" height="1920" alt="train_batch1" src="https://github.com/user-attachments/assets/20144bcb-d514-4489-aa5c-9bce48be8f99" />


## Validation

<img width="1920" height="1920" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/11af791c-9f69-43e1-bdad-bbb22fbc2ee4" />

<img width="1920" height="1920" alt="val_batch0_pred" src="https://github.com/user-attachments/assets/ce8f0afb-053e-434e-acdc-706b9350c636" />

<img width="1920" height="1920" alt="val_batch1_pred" src="https://github.com/user-attachments/assets/168c888a-a627-4104-abdb-c5f2fc7b2449" />


---
## Test

<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/9791f44b-3710-4682-8c68-35e12de97fd6" />

<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/38e68f4e-9861-4344-9d06-a3f9fce6f60a" />

<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/c13fb67d-07f2-4ebc-a401-ba0890543395" />

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/99fea87d-8807-48e5-bfe8-f03e0864f2c0" />

<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/8e10091a-0055-4fd5-ba74-44a8da8e4446" />

<img width="1920" height="1920" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/3343ac23-b215-44eb-9ca9-aedf0a40f1aa" />

<img width="1920" height="1920" alt="val_batch2_labels" src="https://github.com/user-attachments/assets/d80ffd48-4101-4570-8a98-a42be8e1b753" />







