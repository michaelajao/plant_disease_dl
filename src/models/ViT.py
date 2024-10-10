# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

# Import timm for EfficientNetV2
import timm

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds, accuracy_fn  # Custom helper functions
from helper_functions import *

# ================================================================
# Configuration and Settings
# ================================================================
# Set seeds for reproducibility
set_seeds(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32        # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 20        # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Data fractions for training and validation
train_data_fraction = 1.0  # Use 100% of the training data
valid_data_fraction = 1.0  # Use 100% of the validation data

# Device configuration: use multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
print(f"Using device: {device}")

# ================================================================
# Directory Setup
# ================================================================
# Create directories to save results and figures
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../models", exist_ok=True)

# ================================================================
# Data Preparation
# ================================================================
# Specify the path to the dataset
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define image transformations (resize images and convert to tensors)
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

# Load the full training and validation datasets
full_train_data = datasets.ImageFolder(train_dir, transform=transform)
full_valid_data = datasets.ImageFolder(valid_dir, transform=transform)

# Select a subset of the data based on the specified fractions
train_size = int(len(full_train_data) * train_data_fraction)
valid_size = int(len(full_valid_data) * valid_data_fraction)

# Create a subset of the training and validation datasets
np.random.seed(42)  # Ensure reproducibility
train_indices = np.random.choice(len(full_train_data), train_size, replace=False)
valid_indices = np.random.choice(len(full_valid_data), valid_size, replace=False)

train_data = Subset(full_train_data, train_indices)
valid_data = Subset(full_valid_data, valid_indices)

# Data Loaders for training and validation data
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), 
    pin_memory=True, persistent_workers=True
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), 
    pin_memory=True, persistent_workers=True
)

# ================================================================
# Model Definitions
# ================================================================
input_shape = full_train_data[0][0].shape[0]  # Dynamically get input channels
output_size = len(full_train_data.classes)    # Get the number of classes

# ------------------------------
# Model Definitions
# ------------------------------
class ConvNetPlus(nn.Module):
    """
    Defines an improved model with additional layers, batch normalization, and dropout.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(ConvNetPlus, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 2 * (HEIGHT // 4) * (WIDTH // 4),
                      out_features=hidden_units * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_units * 4, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Load EfficientNetV2 with pretrained weights
efficientnetv2_model = timm.create_model(
    'efficientnetv2_rw_s',  # EfficientNetV2 small version
    pretrained=True,
    num_classes=output_size
)

# ================================================================
# Optimizer, Scheduler and AMP Setup
# ================================================================
# Model and optimizer configuration
convnetplus_model = ConvNetPlus(input_shape=input_shape, hidden_units=32, output_shape=output_size)
convnetplus_model = convnetplus_model.to(device)

efficientnetv2_model = efficientnetv2_model.to(device)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizers
convnetplus_optimizer = optim.SGD(convnetplus_model.parameters(), lr=LEARNING_RATE)
efficientnetv2_optimizer = optim.Adam(efficientnetv2_model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler (OneCycleLR for faster convergence)
convnetplus_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer=convnetplus_optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader)
)

efficientnetv2_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer=efficientnetv2_optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader)
)

# AMP Scaler for mixed precision training
scaler = GradScaler()

# ================================================================
# EarlyStopping Class Definition
# ================================================================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# ================================================================
# Training and Evaluation Functions
# ================================================================
def train_model(
    model, optimizer, scheduler, train_loader, valid_loader, device, num_epochs, model_name, optimizer_name
):
    """
    Training loop for the model with mixed precision and early stopping.
    """
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    early_stopping = EarlyStopping(patience=5, min_delta=0.0)
    best_valid_loss, best_epoch, best_model_state = float('inf'), 0, None
    total_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_val_loss / len(valid_loader)
        val_accuracy = (correct_val / total_val) * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        scheduler.step()
        early_stopping(avg_val_loss)
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"../../models/{model_name}_best_model.pth")

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model state
    model.load_state_dict(torch.load(f"../../models/{model_name}_best_model.pth"))

    # Save training history
    history = {
        "epochs": list(range(1, len(train_losses) + 1)),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    return model, history

# ================================================================
# Training and Visualizations
# ================================================================
# Train ConvNetPlus
convnetplus_model, convnetplus_history = train_model(
    model=convnetplus_model,
    optimizer=convnetplus_optimizer,
    scheduler=convnetplus_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="ConvNetPlus",
    optimizer_name="SGD"
)

# Train EfficientNetV2
efficientnetv2_model, efficientnetv2_history = train_model(
    model=efficientnetv2_model,
    optimizer=efficientnetv2_optimizer,
    scheduler=efficientnetv2_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="EfficientNetV2",
    optimizer_name="Adam"
)

# ================================================================
# Plotting and Evaluation
# ================================================================
def plot_loss(history, model_name):
    plt.figure()
    plt.plot(history['epochs'], history['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(history['epochs'], history['val_losses'], label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} Loss over Epochs', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_loss.pdf")
    plt.close()

def plot_accuracy(history, model_name):
    plt.figure()
    plt.plot(history['epochs'], history['train_accuracies'], label='Training Accuracy', linewidth=2)
    plt.plot(history['epochs'], history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    plt.title(f'{model_name} Accuracy over Epochs', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_accuracy.pdf")
    plt.close()

# Plot and save loss and accuracy for each model
plot_loss(convnetplus_history, 'ConvNetPlus')
plot_accuracy(convnetplus_history, 'ConvNetPlus')
plot_loss(efficientnetv2_history, 'EfficientNetV2')
plot_accuracy(efficientnetv2_history, 'EfficientNetV2')

# ================================================================
# Additional Metrics: Precision, Recall, F1
# ================================================================
def compute_metrics(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Computing Metrics", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return precision, recall, f1

# Compute metrics for ConvNetPlus and EfficientNetV2
convnetplus_metrics = compute_metrics(convnetplus_model, valid_loader)
efficientnetv2_metrics = compute_metrics(efficientnetv2_model, valid_loader)

print(f"ConvNetPlus - Precision: {convnetplus_metrics[0]:.2f}, Recall: {convnetplus_metrics[1]:.2f}, F1: {convnetplus_metrics[2]:.2f}")
print(f"EfficientNetV2 - Precision: {efficientnetv2_metrics[0]:.2f}, Recall: {efficientnetv2_metrics[1]:.2f}, F1: {efficientnetv2_metrics[2]:.2f}")

# ================================================================
# End of code
# ================================================================
