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


## Dataset Preparation

The model is trained on a dataset containing images with varying levels of synthetic Gaussian noise. The dataset structure should be as follows:


### Adding Noise to Images

Noise is added dynamically during training using a Gaussian noise function. The `CustomDataset` class handles the loading and augmentation of images by adding random noise based on a sigma value.

Here's a code snippet demonstrating how to implement the dataset preparation:

```python
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Add Gaussian noise
        sigma = np.random.uniform(10, 50)  # Randomly select sigma value for noise
        noisy_image = self.add_gaussian_noise(image, sigma)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, sigma

    def add_gaussian_noise(self, image, sigma):
        """Adds Gaussian noise to an image."""
        image_np = np.array(image) / 255.0  # Normalize the image
        noise = np.random.normal(0, sigma / 255.0, image_np.shape)  # Generate noise
        noisy_image_np = np.clip(image_np + noise, 0, 1)  # Add noise and clip to valid range
        noisy_image = Image.fromarray((noisy_image_np * 255).astype(np.uint8))  # Convert back to PIL Image
        return noisy_image

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example usage
train_dataset = CustomDataset(directory='data/train', transform=transform)
test_dataset = CustomDataset(directory='data/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Number of training images: {len(train_loader.dataset)}")
print(f"Number of testing images: {len(test_loader.dataset)}")
