# 🌿 LeafGuard-CNN

### CNN-Based Crop Disease Detection using EfficientNetV2B3

A deep learning solution for automated crop disease detection across multiple crop types using EfficientNetV2B3 transfer learning.

## Overview

LeafGuard-CNN is a multi-class image classification model designed for detecting and classifying crop diseases. It utilizes transfer learning with EfficientNetV2B3 backbone, combining image preprocessing with two-phase training strategy for optimal performance.

**Supported Crops:** Corn, Cotton, Grape, Potato, Rice, Sugarcane, Tomato, Wheat

## Requirements

- Python 3.8+
- TensorFlow >= 2.10.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- tqdm >= 4.62.0

## Installation

```bash
pip install tensorflow opencv-python numpy tqdm
```

## Dataset Structure

Organize your dataset as follows:

```
Dataset_annam/
├── Corn/
│   ├── Train/
│   │   ├── disease_class_1/
│   │   ├── disease_class_2/
│   │   └── ...
│   └── Valid/
│       ├── disease_class_1/
│       ├── disease_class_2/
│       └── ...
├── Wheat/
│   ├── Train/
│   └── Valid/
└── ... (other crops)
```

## Usage

### 1. Configure Paths

Update the script with your dataset location:

```python
BASE = "/path/to/Dataset_annam"
RAW_TRAIN = os.path.join(BASE, "[CROP_NAME]/Train")
RAW_VAL = os.path.join(BASE, "[CROP_NAME]/Valid")
```

Replace `[CROP_NAME]` with one of: `Corn`, `Cotton`, `Grape`, `Potato`, `Rice`, `Sugarcane`, `Tomato`, `Wheat`

### 2. Run Training

```bash
python train_model.py
```

The script will:
- Preprocess images using CLAHE enhancement
- Apply data augmentation
- Train the classifier head (8 epochs)
- Fine-tune the full network (20 epochs)
- Save the best model as `best_[CROP_NAME]_model.h5`

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | EfficientNetV2B3 (ImageNet pretrained) |
| Input Size | 256×256 RGB images |
| Pooling | Global Average Pooling 2D |
| Dense Layers | 512 units (ReLU) → NUM_CLASSES (Softmax) |
| Dropout | 40% and 30% |

## Training Strategy

**Phase 1: Head Training (8 epochs)**
- Base model frozen
- Learning rate: 1e-3
- Trains only classifier head

**Phase 2: Fine-tuning (20 epochs)**
- Base model unfrozen
- Learning rate: 1e-5
- Fine-tunes entire network

**Optimizer:** Adam with Categorical Crossentropy loss (label smoothing: 0.1)

**Callbacks:**
- ModelCheckpoint: Saves best model
- ReduceLROnPlateau: Reduces LR by 30% if val_loss plateaus
- EarlyStopping: Stops if no improvement for 5 epochs

## Image Preprocessing

1. Resize to 256×256 pixels
2. Apply CLAHE enhancement (clip limit: 2.0, tile grid: 8×8)
3. Data augmentation: rotation, zoom, shift, shear, brightness, flip

## Model Inference

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model('best_corn_model.h5')

# Preprocess image
image = cv2.imread('image.jpg')
image = cv2.resize(image, (256, 256)) / 255.0
image = np.expand_dims(image, axis=0)

# Predict
predictions = model.predict(image)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]
```

## Handling Large Model Files

Model files (.h5) are too large for direct Git commits. Choose one option:

### Option 1: Git LFS
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
```

### Option 2: Cloud Storage
Store models on Google Drive, Hugging Face, or AWS S3 and add download link to documentation.

### Option 3: .gitignore
Add to `.gitignore`:
```
*.h5
```
Document training instructions in README instead of committing models.

## Performance Tips

- Increase epochs in Phase 2 for better convergence
- Reduce batch size from 32 to 16 if memory constrained
- Decrease learning rate further (1e-6) if fine-tuning shows no improvement
- Validate data quality and class balance before training

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Class mismatch error | Ensure Train/Valid have identical disease folders |
| Out of memory | Reduce batch_size from 32 to 16 or 8 |
| Poor validation accuracy | Check data quality, increase augmentation |
| Training plateaus | Lower learning rate or increase EarlyStopping patience |

## Project Structure

```
.
├── train_model.py
├── README.md
├── .gitignore
└── best_[CROP]_model.h5    (stored in cloud)
```

## 📦 Dataset Source

This project uses a **custom-built crop disease dataset** containing high-quality images of various diseased and healthy crop leaves.  
If you require access to the dataset, feel free to **contact us** for collaboration or research purposes.

---

## 👥 Authors

**LeafGuard Team**  
Developed by passionate innovators dedicated to advancing AI in agriculture.

---

## 🌾 Closing Note

Thank you for exploring **LeafGuard-CNN** 🌱  
Our mission is to bridge the gap between **technology and agriculture** by empowering farmers with intelligent disease detection tools.  
We believe that with AI-driven solutions like LeafGuard, the future of **smart and sustainable farming** is closer than ever.

> *"Empowering farmers, one leaf at a time."* 🌿

