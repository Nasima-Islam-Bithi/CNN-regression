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


markdown
Copy code
# Custom Convolutional Neural Network for Image Noise Estimation

This repository contains a PyTorch implementation of a custom Convolutional Neural Network (CNN) model designed for image noise estimation. The model is built upon a modified VGG-16 backbone with added convolutional and regression layers to estimate noise levels in images.

## Model Architecture

The model uses a pre-trained VGG-16 as a feature extractor, followed by additional convolutional layers and a regressor network that predicts the noise level in an input image. Key components include:

- **VGG-16 Feature Extractor**: Uses the first 10 layers of the pre-trained VGG-16.
- **Skip Connections**: Added between specific layers for better gradient flow.
- **Regressor Network**: Fully connected layers used to predict the noise level in images.

### Architecture Details
- Convolutional layers with ReLU and skip connections.
- Regressor with fully connected layers and LeakyReLU activations.
- Layer normalization and dropout for regularization.

### Model Summary
You can print a summary of the model using `torchsummary`.

## Training and Testing

The training process uses mean squared error (MSE) as the loss function and Adam optimizer for weight updates. During training and testing, metrics such as mean absolute error (MAE) and R-squared (R²) score are computed for performance evaluation.

### Key Functions
- **train**: Trains the model and logs metrics such as loss, MAE, and R² to TensorBoard.
- **test**: Evaluates the model on the validation set and logs metrics to TensorBoard.
- **visualize_predictions**: Visualizes input images alongside the true and predicted noise levels for easy inspection of model performance.

### Dataset

The model was trained on images from the [Div2k, Flickr2k, and OST datasets](https://data.vision.ee.ethz.ch/cvl/DIV2K/) with added synthetic noise. The dataset is loaded via the `CustomDataset` class, which applies random noise to the images during data augmentation.

### Example Usage

1. **Training**:
   ```python
   python train.py
Testing:

python
Copy code
python test.py
Model Visualization: You can visualize the model’s predictions using:

python
Copy code
visualize_predictions(model, valloader)
Requirements
To run this project, ensure you have the following dependencies installed:

Python 3.7+
PyTorch
torchvision
scikit-learn
tqdm
TensorBoard
Pillow
matplotlib
Install dependencies using:

bash
Copy code
pip install -r requirements.txt
Sample Results
After training, the model produces the following metrics:

Epoch	Loss	MAE	R²
1	0.0240	0.0879	0.5251
2	0.0031	0.0438	0.9385
10	0.0006	0.0187	0.9884
Visualization

The visualize_predictions function provides a comparison of true and predicted noise levels.

Saving the Model
The trained model can be saved and loaded using:

python
Copy code
torch.save(model.state_dict(), 'noise_estimation_model.pth')
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
PyTorch
TorchVision
TensorBoard
