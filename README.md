# Lung Image Segmentation

## Overview
This project focuses on lung image segmentation using deep learning techniques, specifically U-Net, ResUNet, and RNGUNet. The goal is to accurately segment lung regions from medical images using convolutional neural networks.

## Features
- Implements three deep learning architectures: U-Net, ResUNet, and RNGUNet.
- Uses Dice Coefficient and Dice Loss as key metrics for evaluation.
- Trains models using TensorFlow and Keras.
- Evaluates model performance with accuracy, precision, recall, and F1-score.
- Provides visualization of segmentation results.

## Dependencies
The project requires the following Python libraries:

```python
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
```

## Model Architectures
### 1. U-Net
A widely used architecture for medical image segmentation, consisting of an encoder-decoder structure with skip connections.

Function:
```python
def unet_model(input_size=(256, 256, 1)):
```

### 2. ResUNet
An extension of U-Net incorporating residual connections to improve gradient flow and training stability.

Function:
```python
def ResUNet(input_shape=(256, 256, 1)):
```

### 3. RNGUNet
A modified version of ResUNet incorporating additional enhancements for better segmentation performance.

Function:
```python
def RNGUNet(input_shape=(256, 256, 1)):
```

## Data Preprocessing
- **Load and preprocess images**

```python
def load_data(img_path, mask_path, img_size=(256, 256)):
```

- **Splitting dataset**: Uses `train_test_split` to create training and testing sets.

## Loss Function & Metrics
- **Dice Coefficient** (measures overlap between predicted and ground truth masks)

```python
def dice_coefficient(y_true, y_pred):
```

- **Dice Loss** (1 - Dice Coefficient, used for training optimization)

```python
def dice_loss(y_true, y_pred):
```

## Training
- **Optimization**: Adam optimizer
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Evaluation & Visualization
- **Evaluate models on test set**:

```python
def evaluate_model(model, X_test, Y_test):
```

- **Visualize segmentation results**:

```python
def visualize_results(model, X_test, Y_test, num_images=5):
```

- **Plot training progress**:

```python
def plot_training_progress(history_dict, model_names):
```

- **Compare model performance**:

```python
def plot_model_comparison(metrics_dict):
```

- **Confusion matrices for evaluation**:

```python
def plot_confusion_matrices(y_true, y_preds, model_names, class_labels):
```

- **Create a comparison table**:

```python
def create_comparison_table(metrics_dict):
```

## Results
The project provides segmentation results for different architectures and evaluates their effectiveness based on various metrics.

## Usage
1. Install dependencies using pip:
```sh
pip install tensorflow numpy pandas opencv-python seaborn matplotlib scikit-learn
```
2. Run the notebook to train and evaluate models.

## Conclusion
This project demonstrates deep learning techniques for lung image segmentation and compares different architectures to determine the most effective approach.

