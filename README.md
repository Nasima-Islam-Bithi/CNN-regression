# Custom Noise Estimation using CNN with Skip Connections

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) model for estimating noise in images. The model is based on the pre-trained VGG16 architecture with custom modifications, including additional convolutional layers, skip connections, and a regression head for noise estimation.

## Table of Contents
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Results](#results)
- [Usage](#usage)

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt


Key libraries:

PyTorch
torchvision
scikit-learn
PIL
matplotlib
tqdm
TensorBoard
Model Architecture
The model uses a modified VGG16 as a feature extractor and adds additional convolutional layers with skip connections. The regression head is used to predict the noise level (sigma) in the images.

Key Layers:
Feature extractor: Pretrained VGG16 layers
Convolutional Layers: Additional layers for noise feature extraction
Skip Connections: Bypasses part of the convolution for better gradient flow
Regression Head: Fully connected layers predicting noise level

Data Preparation
The model expects a dataset of images for noise estimation. The dataset is structured in the following format:
data/
  train/
    image1.jpg
    image2.png
    ...
  test/
    image1.jpg
    image2.png
    ...
image Preprocessing
Images are resized to 224x224 and converted to tensors. Noise is added dynamically during training using a Gaussian noise function.

Training the Model
To train the model, use the following script:
python train.py
The training script implements a loop for 10 epochs and logs the metrics to TensorBoard.

Key metrics:

Mean Absolute Error (MAE)
R2 Score
Loss (MSE Loss)
A learning rate scheduler is used to adjust the learning rate after every 5 epochs.

Testing the Model
You can test the model on the validation set using the following command:
python test.py
Results will include the MAE, MSE, R2 score, and loss.

Results
After 10 epochs, the model achieves the following on the test set:

MAE: 0.0136
R2 Score: 0.9936
You can visualize predictions for individual samples using the visualize_predictions function. This will plot the input image and a bar chart comparing the true and predicted noise levels.

Usage
To use the model on your dataset, update the train_dir and val_dir paths in the script and run the training process.

You can also load a pre-trained model using:
model.load_state_dict(torch.load('model.pth'))
Acknowledgments
PyTorch
TorchVision
TensorBoard
