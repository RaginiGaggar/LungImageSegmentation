# Lung Image Segmentation using Deep Learning

## Overview
This project implements lung image segmentation using deep learning techniques, specifically leveraging TensorFlow/Keras. The goal is to segment lung regions from medical images using convolutional neural network (CNN)-based models. Three different models have been implemented and compared to determine the best-performing approach.

## Dataset
The dataset used for this project is the **Montgomery Dataset**, which consists of lung X-ray images and their corresponding segmentation masks.

- **Image Path:** `Montgomery/img/`
- **Mask Path:** `Montgomery/mask/`
- **Image Format:** Grayscale images, resized to `(256, 256)`

## Preprocessing Steps
1. Read images and masks using OpenCV (`cv2`).
2. Resize images and masks to `(256, 256)`.
3. Normalize images to the range `[0, 1]`.
4. Threshold masks to ensure binary segmentation.

## Model Architectures
Three different deep learning models were implemented and compared:

1. **U-Net**: A widely used architecture for medical image segmentation with encoder-decoder layers.
2. **ResUNet**: A variation of U-Net that incorporates residual connections to improve feature propagation and gradient flow.
3. **R2U-Net (Recurrent Residual U-Net)**: An extension of ResUNet that introduces recurrent connections within residual blocks to enhance feature extraction.

### Loss Function & Metrics
- **Dice Coefficient**: Measures overlap between predicted and ground truth masks.
- **Dice Loss**: Used for optimization, defined as `1 - Dice Coefficient`.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

## Training Process
- **Optimizer:** Adam
- **Batch Size:** 16
- **Epochs:** 50
- **Callbacks:**
  - `ModelCheckpoint`: Saves the best model.
  - `EarlyStopping`: Stops training if validation loss does not improve.
  - `ReduceLROnPlateau`: Adjusts learning rate dynamically.
- **GPU Optimization:** TensorFlow is configured to use GPU with memory growth enabled.

## Model Evaluation
The trained models are evaluated on test data using:
- Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix & Classification Report

A comparative analysis of the three models is also included, showing their respective performance metrics.

## How to Run
### Prerequisites
Ensure you have the following installed:
```bash
pip install tensorflow numpy pandas opencv-python matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Clone this repository:
```bash
git clone https://github.com/yourusername/LungImageSeg.git
cd LungImageSeg
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook LungImageSeg.ipynb
```
3. Run all cells to train and evaluate the models.

## Results & Visualization
- The segmented lung images are displayed using Matplotlib.
- Performance metrics are visualized with seaborn.
- Comparison charts show the effectiveness of the three different models.

## Future Improvements
- Implementing advanced architectures like Attention U-Net or Transformer-based segmentation models.
- Using more diverse datasets for better generalization.
- Hyperparameter tuning for improved accuracy.



