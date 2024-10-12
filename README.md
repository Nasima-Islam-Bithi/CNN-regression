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
- [License](#license)
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
