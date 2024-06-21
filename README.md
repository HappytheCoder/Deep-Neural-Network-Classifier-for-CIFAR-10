# Deep Neural Network Classifier for CIFAR-10

This repository contains a script for training a neural network classifier using a custom-defined network architecture and the CIFAR-10 dataset. The script includes functions for model training, validation, and testing, as well as plotting training and validation losses and accuracy over epochs. The trained model is saved as a checkpoint for future use.

## Dependencies

Make sure you have the following dependencies installed before running the script:

- `torch`
- `torchvision`
- `matplotlib`
- `platform`

## Usage

1. **Ensure all dependencies are installed.**

2. **Run the script**:
   
   The script will:
   - Train the model
   - Plot the training and validation losses
   - Save the trained model as a checkpoint (`checkpoint.pth`)
     
## Script Overview

The script performs the following steps:

1. **Import Dependencies**:
   - Libraries such as `torch`, `torchvision`, `matplotlib`, and `platform` are imported.

2. **Set Device for Training**:
   - The script checks the operating system and sets the device to MPS for MacOS, CUDA for other platforms if available, or CPU otherwise.

3. **Define the Custom Neural Network**:
   - A custom neural network class `Network` is defined, inheriting from `nn.Module`.

4. **Define Transformations**:
   - Data transformations for training and testing are defined using `transforms.Compose`.

5. **Create Datasets and DataLoaders**:
   - The CIFAR-10 dataset is downloaded and loaded into training and testing dataloaders.

6. **Load Pre-trained ResNet18 Model**:
   - A pre-trained ResNet18 model is loaded and its parameters are frozen.
   - The classifier is replaced with the custom-defined network.

7. **Specify Loss Function and Optimizer**:
   - The loss function is set to `nn.NLLLoss`.
   - The optimizer is set to `optim.Adam`.

8. **Train the Model**:
   - The model is trained for a specified number of epochs, with training and validation losses recorded.

9. **Plot Training and Validation Losses/Accuracy**:
   - Training and validation losses and accuracy are plotted over epochs.

10. **Test the Model**:
    - The model is tested on the test dataset and the accuracy is computed.

11. **Save the Model Checkpoint**:
    - The trained model is saved as a checkpoint for future use.

## Model Checkpoint

The checkpoint includes:
- Input size
- Output size
- Hidden layers
- Dropout probability
- Model state dictionary

The checkpoint is saved as `checkpoint.pth`.
