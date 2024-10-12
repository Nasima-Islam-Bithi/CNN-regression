markdown
Copy code
# Custom Convolutional Neural Network for Image Noise Estimation

This repository contains a PyTorch implementation of a custom Convolutional Neural Network (CNN) model designed for image noise estimation. The model is based on a modified VGG-16 backbone with additional convolutional and regression layers to estimate noise levels in images.

## Model Architecture

The model utilizes:
- **VGG-16 Feature Extractor**: Pre-trained VGG-16 model as the backbone, up to the first 10 layers.
- **Skip Connections**: Added between certain layers to enhance gradient flow.
- **Additional Layers**: Extra convolutional layers after the VGG-16 part.
- **Regressor Network**: Fully connected layers designed to predict the noise level.

### Model Summary

- Convolutional layers with ReLU activations and skip connections.
- Fully connected layers with LeakyReLU for the regression task.
- Dropout and Layer Normalization for regularization.

You can print a summary of the model using the `torchsummary` package.

## Training and Testing

The training process uses the **Mean Squared Error (MSE)** loss function and **Adam optimizer** for updating weights. During training and testing, the following metrics are computed:
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

Metrics are logged to TensorBoard for visualization.

### Key Functions

- **train**: Trains the model and logs metrics (loss, MAE, R²) to TensorBoard.
- **test**: Evaluates the model and logs the same metrics on the validation set.
- **visualize_predictions**: Visualizes model predictions on validation data with true noise levels.

### Dataset

The dataset used for training includes images from the [Div2k, Flickr2k, and OST datasets](https://data.vision.ee.ethz.ch/cvl/DIV2K/), augmented with synthetic noise. The custom `CustomDataset` class handles data loading and random noise generation.

## Example Usage

1. **Training**:
   ```bash
   python train.py
Testing:

bash
Copy code
python test.py
Visualization: You can visualize the model's predictions on validation images using:

python
Copy code
visualize_predictions(model, valloader)
Sample Results
After training, here are the key results:

Epoch	Loss	MAE	R²
1	0.0240	0.0879	0.5251
2	0.0031	0.0438	0.9385
10	0.0006	0.0187	0.9884
Installation
To get started with this project, install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Requirements
Python 3.7+
PyTorch
torchvision
scikit-learn
tqdm
TensorBoard
Pillow
matplotlib
Saving and Loading the Model
You can save the trained model using:

python
Copy code
torch.save(model.state_dict(), 'noise_estimation_model.pth')
To load the model for further inference or training:

python
Copy code
model.load_state_dict(torch.load('noise_estimation_model.pth'))
model.eval()  # Set the model to evaluation mode
Visualization
You can visualize the noise predictions against true values with:

python
Copy code
visualize_predictions(model, valloader)

This function compares input images, true noise levels, and predicted noise levels.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
The pre-trained VGG-16 model is provided by torchvision. The noise estimation method is inspired by recent advancements in image quality and noise analysis.

sql
Copy code

You can paste this directly into your `README.md` file. It covers all essential parts like model architecture, training, usage, requirements, and more, formatted cleanly for a Markdown file.





