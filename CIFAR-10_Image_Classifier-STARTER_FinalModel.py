"""
This script trains a neural network classifier using a custom-defined network architecture
and CIFAR-10 dataset. It includes functions for model training, validation, and testing, as
well as plotting training and validation losses and accuracy over epochs. The trained model
is saved as a checkpoint for future use.


Dependencies:
    - torch
    - torchvision
    - matplotlib
    - platform

Usage:
    - Ensure all dependencies are installed.
    - Run the script. It will train the model, plot the training and validation losses,
      and save the trained model as a checkpoint.
"""

# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import platform


# Set device for training
if platform.system() == 'Darwin':  # Check if the system is MacOS
    mp_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
else:  # For other platforms like Windows or Linux
    mp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    """
    Define a custom neural network for the classifier.
    """

    def __init__(self, input, output, hidden_layer_PEs, drop_p):
        """
        Initialize the neural network.

        Args:
            input (int): Number of input features.
            output (int): Number of output features.
            hidden_layer_PEs (list): List of number of neurons in each hidden layer.
            drop_p (float): Dropout probability.
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input, hidden_layer_PEs[0])])
        if len(hidden_layer_PEs) > 1:
            for i in range(1, len(hidden_layer_PEs)):
                self.hidden_layers.append(nn.Linear(hidden_layer_PEs[i - 1], hidden_layer_PEs[i]))
        self.out = nn.Linear(hidden_layer_PEs[-1], output)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        x = x.view(x.shape[0], -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = F.log_softmax(self.out(x), dim=1)
        return x


# Define transformations for training and testing data
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),  # Rotate images randomly by a maximum of 30 degrees
    transforms.RandomResizedCrop(224),  # Crop and resize images to 224x224 randomly
    transforms.RandomHorizontalFlip(),  # Flip images horizontally randomly
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize images with mean and standard deviation
])

test_transforms = transforms.Compose([
    transforms.Resize(255),  # Resize images to 255x255
    transforms.CenterCrop(224),  # Crop images at the center to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize images with mean and standard deviation
])

# Create training set and define training dataloader
train_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create test set and define test dataloader
test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# The 10 classes in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier with our own network
input = model.fc.in_features
output = len(classes)
hidden_layer_PEs = [256]
p_drop = 0.2
model.fc = Network(input, output, hidden_layer_PEs, p_drop)

# Move model to device
model.to(mp_device)

# Specify a loss function and an optimizer
criterion = nn.NLLLoss()
lr = 0.001
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

# Train the neural network model and record the average loss at each epoch
epochs = 10
train_loss = []
validation_loss = []
accuracy = []
for e in range(epochs):
    running_loss = 0
    running_valid_loss = 0
    running_accuracy = 0
    model.train()
    for image, label in train_dataloader:
        image, label = image.to(mp_device), label.to(mp_device)
        optimizer.zero_grad()
        logits = model.forward(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        with torch.no_grad():
            model.eval()
            for imageT, labelT in test_dataloader:
                imageT, labelT = imageT.to(mp_device), labelT.to(mp_device)
                logitsT = model.forward(imageT)
                p = torch.exp(logitsT)
                top_p, top_class = p.topk(1, dim=1)
                equals = top_class == labelT.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor))
                lossT = criterion(logitsT, labelT)
                running_valid_loss += lossT.item()

        train_loss.append((running_loss / len(train_dataloader)))
        validation_loss.append((running_valid_loss / len(test_dataloader)))
        accuracy.append((running_accuracy / len(test_dataloader)))
        # Print training loss for the epoch
        print(f"Epoch: {e + 1}/{epochs}, Training Loss: {running_loss / len(train_dataloader):.3f}")
        print(f"Epoch: {e + 1}/{epochs}, Validation Loss: {running_valid_loss / len(test_dataloader):.3f}")
        # Print accuracy
        print(f"Epoch: {e + 1}/{epochs}, Accuracy: {running_accuracy.item() * 100 / len(test_dataloader):.3f}%")

# Plot the training loss and validation loss/accuracy
fig, ax1 = plt.subplots()

# Create a list of epochs
epochs_list = list(range(1, epochs + 1))

# Plotting the training loss and validation loss on the primary y-axis
ax1.plot(epochs_list, train_loss, label='Training Loss', color='blue')
ax1.plot(epochs_list, validation_loss, label='Validation Loss', color='red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

# Create a secondary y-axis for accuracy
ax2 = ax1.twinx()

# Plotting the validation accuracy on the secondary y-axis
ax2.plot(epochs_list, accuracy, label='Validation Accuracy', color='green')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Display the plot
plt.show()

# Testing the model
with torch.no_grad():
    imageT, labelT = next(iter(test_dataloader))
    imageT, labelT = imageT.to(mp_device), labelT.to(mp_device)
    logitsT = model(imageT)
    p = torch.exp(logitsT)
    top_p, top_class = p.topk(1, dim=1)
    equals = top_class == labelT.view(*top_class.shape)
    percent_correct_prediction = torch.mean(equals.type(torch.FloatTensor)) * 100

# Save the model checkpoint
checkpoint = {'input_size': input,
              'output_size': output,
              'hidden_layers': [each.out_features for each in model.fc.hidden_layers],
              'drop_out' : p_drop,
              'state_dict': model.state_dict()}
print(checkpoint)

torch.save(checkpoint, 'checkpoint.pth')  # Save the checkpoint
