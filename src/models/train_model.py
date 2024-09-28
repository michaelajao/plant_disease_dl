# Import necessary libraries
import os
import sys
import time  # For measuring the time of each epoch and total training time
import pandas as pd  # For saving results in a DataFrame
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
from helper_functions import (
    set_seeds,
    accuracy_fn,
)  # Custom helper functions for seed setting and accuracy calculation

# Set the seed for reproducibility
set_seeds(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32  # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 10  # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Data fractions for training and validation
train_data_fraction = 0.3  # Use 30% of the training data
valid_data_fraction = 0.1  # Use 10% of the validation data

# Check if CUDA is available, and set the device to GPU if available, else CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Specify the path to the dataset
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define image transformations (resize images and convert to tensors)
transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
    ]
)

# Load the full training and validation datasets
full_train_data = datasets.ImageFolder(train_dir, transform=transform)
full_valid_data = datasets.ImageFolder(valid_dir, transform=transform)

# Select a subset of the data based on the specified fractions
train_size = int(len(full_train_data) * train_data_fraction)
valid_size = int(len(full_valid_data) * valid_data_fraction)

# Create a subset of the training and validation datasets
train_indices = np.random.choice(len(full_train_data), train_size, replace=False)
valid_indices = np.random.choice(len(full_valid_data), valid_size, replace=False)

train_data = Subset(full_train_data, train_indices)
valid_data = Subset(full_valid_data, valid_indices)

# Data Loaders for training and validation data
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)


# Baseline Model definition
class BaselineModel(nn.Module):
    """
    Defines a simple baseline model with fewer layers.

    Args:
        input_shape (int): Number of input channels.
        hidden_units (int): Number of units in the hidden layers.
        output_shape (int): Number of output classes.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(BaselineModel, self).__init__()
        # Define convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * (HEIGHT // 2) * (WIDTH // 2),
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        x = self.conv_block(x)  # Pass input through convolutional layers
        x = self.classifier(x)  # Pass through fully connected layers
        return x


# ConvNetPlus Model definition (formerly ImprovedModel)
class ConvNetPlus(nn.Module):
    """
    Defines an improved model with additional layers, batch normalization, and dropout.

    Args:
        input_shape (int): Number of input channels.
        hidden_units (int): Number of units in the hidden layers.
        output_shape (int): Number of output classes.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(ConvNetPlus, self).__init__()
        # Define convolutional layers with Batch Normalization and Dropout
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),  # Dropout to prevent overfitting
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # Define fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 2 * (HEIGHT // 4) * (WIDTH // 4),
                out_features=hidden_units * 4,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_units * 4, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# TinyVGG Model definition
class TinyVGG(nn.Module):
    """
    Defines a TinyVGG model.

    Args:
        input_shape (int): Number of input channels.
        hidden_units (int): Number of units in the hidden layers.
        output_shape (int): Number of output classes.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units * 2,
                out_channels=hidden_units * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 2 * (HEIGHT // 4) * (WIDTH // 4),
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# Instantiate the models
output_size = len(full_train_data.classes)  # Get the number of classes
baseline_model = BaselineModel(
    input_shape=3, hidden_units=8, output_shape=output_size
).to(device)
convnetplus_model = ConvNetPlus(
    input_shape=3, hidden_units=16, output_shape=output_size
).to(device)
tinyvgg_model = TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size).to(
    device
)

# Define loss function and optimizer for all models
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

# Define optimizers with different types
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
convnetplus_optimizer = optim.SGD(convnetplus_model.parameters(), lr=LEARNING_RATE)
tinyvgg_optimizer = optim.RMSprop(tinyvgg_model.parameters(), lr=LEARNING_RATE)


# Function to train the model for one epoch (now includes accuracy)
def train_model(model, train_loader, optimizer, loss_fn, accuracy_fn, epoch):
    """
    Trains the model for one epoch and returns the average training loss and accuracy.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_fn (function): Function to compute accuracy.
        epoch (int): The current epoch number.

    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    train_loss, train_acc = 0, 0  # Initialize the loss and accuracy accumulators

    for batch, (X, y) in enumerate(
        tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)
    ):
        X, y = X.to(device), y.to(device)  # Move data to the selected device

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Calculate accuracy
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    avg_train_loss = train_loss / len(train_loader)  # Calculate average loss
    avg_train_acc = train_acc / len(train_loader)  # Calculate average accuracy
    return avg_train_loss, avg_train_acc


# Function to evaluate the model on validation data
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    """
    Evaluates the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): The data loader for validation data.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_fn (function): Function to compute accuracy.

    Returns:
        dict: A dictionary with validation loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    valid_loss, valid_acc = 0, 0  # Initialize accumulators for loss and accuracy

    with torch.no_grad():  # Disable gradient calculation
        for X, y in tqdm(data_loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)  # Move data to the selected device

            # Forward pass
            y_pred = model(X)
            valid_loss += loss_fn(y_pred, y).item()  # Accumulate validation loss
            valid_acc += accuracy_fn(
                y_true=y, y_pred=y_pred.argmax(dim=1)
            )  # Compute accuracy

    avg_valid_loss = valid_loss / len(data_loader)  # Calculate average validation loss
    avg_valid_acc = valid_acc / len(
        data_loader
    )  # Calculate average validation accuracy

    return {"model_loss": avg_valid_loss, "model_acc": avg_valid_acc}


# Function to train and evaluate the model over multiple epochs (includes early stopping)
def train_and_evaluate(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_fn,
    accuracy_fn,
    num_epochs,
    model_name,
    optimizer_name,
    early_stopping_patience=1,
):
    """
    Trains and evaluates the model for a specified number of epochs with early stopping.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_fn (function): Function to compute accuracy.
        num_epochs (int): The maximum number of epochs for training.
        model_name (str): Name of the model (for logging purposes).
        optimizer_name (str): Name of the optimizer used.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        dict: A dictionary containing best metrics and total training time.
    """
    results = []
    total_start_time = time.time()  # Start time for the full training

    best_valid_acc = 0
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of the epoch
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train the model and print training loss and accuracy
        avg_train_loss, avg_train_acc = train_model(
            model, train_loader, optimizer, loss_fn, accuracy_fn, epoch
        )
        print(f"Train loss: {avg_train_loss:.4f}, Train accuracy: {avg_train_acc:.2f}%")

        # Evaluate the model and print validation loss and accuracy
        valid_results = eval_model(model, valid_loader, loss_fn, accuracy_fn)
        print(
            f"Validation loss: {valid_results['model_loss']:.4f}, Validation accuracy: {valid_results['model_acc']:.2f}%"
        )

        # Calculate and display the time taken for the epoch
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        print("-" * 50)

        # Save metrics
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "valid_loss": valid_results["model_loss"],
            "valid_acc": valid_results["model_acc"],
            "epoch_duration": epoch_duration,
            "model": model_name,
            "optimizer": optimizer_name,
        }
        results.append(epoch_result)

        # Early Stopping based on validation accuracy
        if valid_results["model_acc"] > best_valid_acc:
            best_valid_acc = valid_results["model_acc"]
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping triggered after {early_stopping_patience} epoch(s) with no improvement."
            )
            break

    total_training_time = time.time() - total_start_time  # Total time for training
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Return best results
    best_result = {
        "model": model_name,
        "optimizer": optimizer_name,
        "best_epoch": best_epoch,
        "best_valid_loss": results[best_epoch - 1]["valid_loss"],
        "best_valid_acc": best_valid_acc,
        "total_training_time": total_training_time,
    }
    return results, best_result


# Start training and evaluating the baseline model
print("Training Baseline Model")
baseline_results, baseline_best = train_and_evaluate(
    model=baseline_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=baseline_optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    num_epochs=NUM_EPOCHS,
    model_name="BaselineModel",
    optimizer_name="Adam",
    early_stopping_patience=1,  # Early stopping patience set to 1
)

# Start training and evaluating the ConvNetPlus model
print("\nTraining ConvNetPlus Model")
convnetplus_results, convnetplus_best = train_and_evaluate(
    model=convnetplus_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=convnetplus_optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    num_epochs=NUM_EPOCHS,
    model_name="ConvNetPlus",
    optimizer_name="SGD",
    early_stopping_patience=1,  # Early stopping patience set to 1
)

# Start training and evaluating the TinyVGG model
print("\nTraining TinyVGG Model")
tinyvgg_results, tinyvgg_best = train_and_evaluate(
    model=tinyvgg_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=tinyvgg_optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    num_epochs=NUM_EPOCHS,
    model_name="TinyVGG",
    optimizer_name="RMSprop",
    early_stopping_patience=1,  # Early stopping patience set to 1
)

# Combine the best results
best_results_df = pd.DataFrame([baseline_best, convnetplus_best, tinyvgg_best])

# Display the best results DataFrame for comparison
print("\nBest Results for Each Model:")
print(best_results_df)

# Save the combined best results to a single CSV file
best_results_df.to_csv("model_best_results.csv", index=False)
