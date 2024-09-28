# Importing necessary libraries
import os
import sys
import time  # For measuring the time of each epoch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Load helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds, accuracy_fn

# Set the seed for reproducibility
set_seeds(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Smaller learning rate for better convergence
NUM_EPOCHS = 5
HEIGHT, WIDTH = 224, 224

# Fraction of data to use for training and validation
train_data_fraction = 0.3  # Use 30% of the training data
valid_data_fraction = 0.1  # Use 10% of the validation data

# Check if CUDA is available and configure accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to the dataset
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
    ]
)

# Load the full training and validation datasets
full_train_data = datasets.ImageFolder(train_dir, transform=transform)
full_valid_data = datasets.ImageFolder(valid_dir, transform=transform)

# Select a subset of the data based on the desired fraction
train_size = int(len(full_train_data) * train_data_fraction)
valid_size = int(len(full_valid_data) * valid_data_fraction)

# Create a subset of the training and validation datasets
train_data = Subset(
    full_train_data, np.random.choice(len(full_train_data), train_size, replace=False)
)
valid_data = Subset(
    full_valid_data, np.random.choice(len(full_valid_data), valid_size, replace=False)
)

# Data Loaders
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


# Model Definitions
class ImprovedModel(nn.Module):
    """
    Defines an improved baseline model with convolutional and fully connected layers.
    """

    def __init__(self, output_size: int):
        super(ImprovedModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (HEIGHT // 4) * (WIDTH // 4), 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Model Instantiation
output_size = len(full_train_data.classes)
model = ImprovedModel(output_size).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Function to train the model
def train_model(model, train_loader, optimizer, loss_fn, epoch):
    """
    Trains the model for one epoch and returns the average training loss.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss_fn (torch.nn.Module): The loss function.
        epoch (int): The current epoch number.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(
        tqdm(train_loader, desc=f"Training epoch {epoch+1}")
    ):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


# Function to evaluate the model
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    """
    Evaluates the model on the given data_loader.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): The data loader for validation data.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_fn: Function to compute accuracy.

    Returns:
        dict: A dictionary with validation loss and accuracy.
    """
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            valid_loss += loss_fn(y_pred, y).item()
            valid_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    avg_valid_loss = valid_loss / len(data_loader)
    avg_valid_acc = valid_acc / len(data_loader)

    return {"model_loss": avg_valid_loss, "model_acc": avg_valid_acc}


# Function to train and evaluate the model across epochs
def train_and_evaluate(
    model, train_loader, valid_loader, optimizer, loss_fn, accuracy_fn, num_epochs
):
    """
    Trains and evaluates the model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_fn: Function to compute accuracy.
        num_epochs (int): The number of epochs for training.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train the model
        avg_train_loss = train_model(model, train_loader, optimizer, loss_fn, epoch)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Evaluate the model
        valid_results = eval_model(model, valid_loader, loss_fn, accuracy_fn)
        print(
            f"Validation loss: {valid_results['model_loss']:.4f}, Validation accuracy: {valid_results['model_acc']:.2f}%"
        )

        # Time taken for the epoch
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        print("-" * 50)


# Start training and evaluation
train_and_evaluate(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    num_epochs=NUM_EPOCHS,
)
