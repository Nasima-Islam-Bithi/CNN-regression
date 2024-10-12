# Custom Convolutional Neural Network for Image Noise Estimation

This repository contains a PyTorch implementation of a custom Convolutional Neural Network (CNN) designed to estimate the noise level (sigma) in images. The model leverages a pre-trained VGG-16 backbone with additional convolutional layers, skip connections, and a regression head to accurately predict the noise intensity present in input images.

## Table of Contents

- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Visualization](#visualization)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Features

- **Pre-trained VGG-16 Backbone**: Utilizes the first 10 layers of VGG-16 for feature extraction.
- **Custom Convolutional Layers**: Additional layers with ReLU activations and skip connections to enhance feature learning.
- **Regression Head**: Fully connected layers to predict the noise level in images.
- **Data Augmentation**: Dynamically adds Gaussian noise to images during training for robust learning.
- **Performance Metrics**: Tracks Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) scores.
- **Visualization**: Tools to visualize predictions against true noise levels.
- **TensorBoard Integration**: Logs training and testing metrics for real-time monitoring.

## Model Architecture

The model architecture is based on a modified VGG-16 network with additional convolutional layers and a regression head. Here's a breakdown of the components:

- **VGG-16 Feature Extractor**: Uses the first 10 layers of the pre-trained VGG-16 model to extract high-level features from input images.
- **Additional Convolutional Layers**: 
  - `conv_1`: Two convolutional layers increasing and then reducing the channel dimensions.
  - `conv_2` & `conv_3`: Additional layers for processing and skip connections.
- **Skip Connections**: Facilitates better gradient flow and feature reuse by adding the output of `conv_3` to `conv_2`.
- **Regressor Network**: A series of fully connected layers with LeakyReLU activations, Layer Normalization, and Dropout for predicting the noise level.

### Model Summary

You can view a summary of the model architecture using the `torchsummary` package:

```python
from torchsummary import summary
summary(model, input_size=(3, 224, 224), device='cuda')
```


## Dataset Preparation
The model is trained on a dataset containing images with varying levels of synthetic Gaussian noise. The dataset structure should be as follows:
data/
  train/
    image1.jpg
    image2.png
    ...
  test/
    image1.jpg
    image2.png
    ...
### Adding Noise to Images
Noise is added dynamically during training using a Gaussian noise function. The CustomDataset class handles the loading and augmentation of images by adding random noise based on a sigma value.

## Installation
Clone the repository and install the required dependencies
### Requirements
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- Pillow
- matplotlib
- tqdm
- TensorBoard

## Usage
### Training
To train the model, ensure your training and validation datasets are properly structured and update the train_dir and val_dir paths in your training script.
Hyperparameters are used as follows:
- Optimizer: Adam optimizer with a learning rate of 0.0001.
- Loss Function: Mean Squared Error (MSE) Loss.
- Scheduler: Learning rate scheduler that decays the learning rate by a factor of 0.1 every 5 epochs.
- Epochs: Configured for 10 epochs.
- Logging: Metrics are logged to TensorBoard for monitoring.

### Testing
After training, evaluate the model on the validation set.
- Computes loss, MAE, MSE, and R² scores.
- Logs metrics to TensorBoard.
- Prints out the performance metrics.

### Visualization
Visualize the model's predictions compared to the true noise levels using the visualize_predictions function.
- Displays input images.
- Shows bar charts comparing true sigma, predicted sigma, and estimated sigma using scikit-learn's estimate_sigma.


## Results

After training for 10 epochs, the model achieves impressive performance on the test set:

| Epoch |   Loss   |   MAE   |   MSE   |   R²   |
|-------|----------|---------|---------|--------|
|   1   | 0.0240   | 0.0879  | 0.0088  | 0.5251 |
|   2   | 0.0031   | 0.0438  | 0.0030  | 0.9428 |
|   3   | 0.0024   | 0.0379  | 0.0030  | 0.9405 |
|   4   | 0.0018   | 0.0326  | 0.0018  | 0.9649 |
|   5   | 0.0013   | 0.0283  | 0.0009  | 0.9812 |
|   6   | 0.0007   | 0.0206  | 0.0005  | 0.9912 |
|   7   | 0.0006   | 0.0196  | 0.0004  | 0.9926 |
|   8   | 0.0006   | 0.0194  | 0.0004  | 0.9925 |
|   9   | 0.0006   | 0.0189  | 0.0005  | 0.9909 |
|  10   | 0.0006   | 0.0187  | 0.0003  | 0.9936 |

The model consistently improves its performance across epochs, achieving a final R² score of 0.9936, indicating a high level of accuracy in noise estimation.

## Saving and Loading the Model
### Saving the Model
After training, save the model's state dictionary:
```python
torch.save(model.state_dict(), 'noise_estimation_model.pth')
```

### Loading the Model
To load the saved model for inference or further training:
```python
model = CustomCNN()
model.load_state_dict(torch.load('noise_estimation_model.pth'))
model.to(device)
model.eval()  # Set to evaluation mode
```

## Acknowledgments
- PyTorch: An open-source machine learning library for Python.
- TorchVision: PyTorch's computer vision library providing datasets, model architectures, and image transformations.
- VGG-16: The pre-trained VGG-16 model used as the backbone for feature extraction.
- Scikit-learn: Utilized for computing performance metrics like MAE and R².
- TensorBoard: For logging and visualizing training and testing metrics.
- Research Inspirations: This project is inspired by advancements in image quality assessment and noise estimation techniques in computer vision.

