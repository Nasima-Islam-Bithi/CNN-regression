# Prediction of Gaussian noise level in RGB image Using Deep CNN

This project focuses on training a deep learning model to predict noise levels in RGB images using a Custom Convolutional Neural Network (CNN). The model processes images, adds Gaussian noise, and then trains to estimate the noise parameter (sigma) applied to the image. The project uses PyTorch for model development and TensorBoard for tracking metrics.

#Project Structure
    Model Architecture: The architecture is based on a Custom CNN that processes images and predicts the noise sigma.
    Data: The images are loaded from a dataset and Gaussian noise is added with different sigma values.
    Metrics: The model performance is evaluated using MAE (Mean Absolute Error), MSE (Mean Squared Error), and R² score.
    Logging: TensorBoard is used for tracking the loss and evaluation metrics over the training process.

Prerequisites
    Python 3.8 or higher
    Libraries:
        PyTorch
        Torchvision
        NumPy
        PIL (Python Imaging Library)
        Scikit-learn
        TensorBoard

Install the necessary libraries using:
pip install torch torchvision numpy pillow scikit-learn tensorboard

Running the Notebook
    Data Preparation:
        Place your image datasets (e.g., DIV2K, Flickr2k, OST) in the appropriate folders.
        The images will be loaded, resized, and transformed using torchvision.transforms.

    Training the Model:
        The training loop includes adding Gaussian noise to the images, followed by a forward pass through the CNN.
        Training metrics such as MAE, MSE, and R² are logged using TensorBoard.
        Run the training cells to start the process.

    Evaluating the Model:
        After training, the model is evaluated using a validation dataset to check its accuracy in predicting the noise sigma.
        The notebook includes cells to compute the final MAE, MSE, and R² for the validation set.

    Visualization:
        TensorBoard is used to visualize the training progress and evaluation metrics.
        To start TensorBoard, run:
        tensorboard --logdir=runs

Custom CNN Architecture

The Custom CNN model is designed to estimate the noise level in an image (sigma) with the following layers:

    Convolutional layers with ReLU activations and max-pooling
    A fully connected regression network that outputs the noise estimate

Usage

    Clone the repository:
    git clone <repository-link>
    cd <repository-folder>

    Launch the Jupyter Notebook:
    jupyter notebook noise.ipynb

Follow the instructions in the notebook to start training and evaluating the model.
