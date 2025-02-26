# Lung Image Segmentation

## Overview
This project focuses on lung image segmentation using deep learning techniques, leveraging U-Net, ResUNet, and RNGUNet architectures. The goal is to achieve precise segmentation of lung regions from medical images using convolutional neural networks (CNNs).

## Key Features
- **Three powerful architectures**: U-Net, ResUNet, and RNGUNet for segmentation.
- **Performance-driven metrics**: Dice Coefficient and Dice Loss for evaluation.
- **Optimized training**: TensorFlow and Keras with Adam optimizer and adaptive learning strategies.
- **Comprehensive evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization tools**: Segmentation overlays, model performance plots, and comparative analyses.

## Dependencies
Ensure the following Python libraries are installed:

```python
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
```

## Model Architectures
### 1. **U-Net**
A widely adopted encoder-decoder network with skip connections to retain spatial information.

### 2. **ResUNet**
An advanced U-Net variant incorporating residual connections for improved gradient flow and training stability.

### 3. **RNGUNet**
A refined ResUNet model with enhanced features for superior segmentation performance.

## Data Processing
- **Image & mask preprocessing**: Standardized resizing and normalization.
- **Dataset split**: `train_test_split` ensures an effective training and testing split.

## Loss Function & Metrics
- **Dice Coefficient**: Measures overlap between predictions and ground truth masks.
- **Dice Loss**: Optimized training with `1 - Dice Coefficient`.

## Training Strategy
- **Optimizer**: Adam with fine-tuned learning rate adjustments.
- **Callbacks**: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau for stable convergence.

## Evaluation & Visualization
- **Performance Metrics**: Accuracy, precision, recall, F1-score.
- **Confusion Matrices**: Visual representation of model predictions.
- **Segmentation Results**: Side-by-side comparisons of input, ground truth, and model predictions.
- **Training Progress**: Plot loss and metric trends over epochs.
- **Model Comparison**: Performance comparison across different architectures.

## Results
This project systematically evaluates and visualizes the effectiveness of different segmentation models, offering insights into their performance.

## Usage
1. Install dependencies:
```sh
pip install tensorflow numpy pandas opencv-python seaborn matplotlib scikit-learn
```
2. Run the notebook to train and evaluate the models.

## Conclusion
This study demonstrates how deep learning can effectively segment lung regions from medical images, providing a comparative analysis of various architectures to determine the most effective approach.

