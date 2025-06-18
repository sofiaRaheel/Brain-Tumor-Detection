# 🧠 Brain Tumor Classification from MRI Scans

## Project Overview

This project builds a deep learning system to classify brain tumors from MRI scans into four clinical categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

It compares two neural network architectures:
- A **custom Convolutional Neural Network (CNN)** built specifically for medical image analysis.
- A **ResNet18-based model** using transfer learning for improved performance.

The project integrates **Grad-CAM visualizations** for explainability, enabling medical practitioners to see what regions influenced the model's predictions.

---

## Key Features

- **Accurate Classification**: Distinguishes among four tumor types with high accuracy.
- **Dual Architecture Approach**:
  - Custom CNN: Designed for nuanced medical imaging patterns.
  - Modified ResNet18: Leverages pretrained weights with fine-tuned adaptation.
- **Explainable AI**: Grad-CAM heatmaps reveal decision-critical areas in MRI scans.
- **Robust Training Techniques**:
  - Class imbalance handled with weighted loss functions.
  - Extensive data augmentation to simulate real-world imaging variability.
  - Cosine annealing and early stopping for stable convergence.

---

## Dataset Information

The dataset comprises **5,862 MRI scans**, preprocessed to **224×224 resolution** and normalized with ImageNet statistics. Images span multiple orientations (axial, coronal, sagittal) and scanner types.

- **Training Set**: 4,551 images (77.6%)
  - Glioma: 826
  - Meningioma: 822
  - Pituitary: 827
  - No Tumor: 395

- **Test Set**: 1,311 images (22.4%) with similar class distribution.

---

## Model Architectures

### 1. Custom CNN Model

**Design Philosophy**:
- Layered convolution blocks with batch normalization.
- Dropout for regularization.
- Tailored capacity for complex medical imaging.

**Training Details**:
- He-normal initialization
- 50 epochs with early stopping
- Optimizer: AdamW with weight decay (0.01)
- Class-weighted loss function
- Cosine annealing LR (0.0001 → 1e-6)

**Performance**:
- **Test Accuracy**: 91.2%
- Balanced precision, recall, and F1-score

---

### 2. Enhanced ResNet18 Model

**Modifications**:
- Pretrained on ImageNet; early layers frozen.
- Final convolutional blocks fine-tuned.
- Custom classifier:
  - Dense (512 units) + BatchNorm
  - 60% dropout
  - 4-unit output layer

**Training Details**:
- Faster convergence (30 epochs)
- Maintains pretrained feature extraction
- Avoids catastrophic forgetting

**Performance**:
- **Test Accuracy**: 96.4%

---

## 🛠️ Technical Implementation

### 🔄 Data Pipeline

- Random resized crop (224×224 from 256×256)
- Horizontal flip (50%), vertical flip (30%)
- ±30° rotation
- Color jitter (20% brightness/contrast)
- Gaussian blur (3×3)
- Normalization using ImageNet stats

### 🧪 Training Optimization

- **Loss Function**: Weighted cross-entropy (inverse class frequency)
- **Learning Rate**: Cosine annealing
- **Early Stopping**: Patience = 5
- **Hardware**: Trained using NVIDIA GPU acceleration

---

## Model Interpretation

Grad-CAM visualizations:
- Highlight key image regions influencing predictions.
- Ensure focus on anatomically relevant areas.
- Help detect and reduce model biases.
- Improve trust and transparency for clinical adoption.

---

## Performance Comparison

| Metric         | Custom CNN | ResNet18 |
|----------------|------------|----------|
| **Accuracy**   | 91.2%      | 96.4%    |
| **Precision**  | 0.91       | 0.96     |
| **Recall**     | 0.90       | 0.96     |
| **F1-Score**   | 0.90       | 0.96     |
| **Train Time** | 2h 15m     | 1h 10m   |

---

## Usage

### Training

```bash
python train.py --model [cnn|resnet] --epochs 50 --data_dir ./Brain_Tumor_Data
