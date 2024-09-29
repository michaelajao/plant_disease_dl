# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Hyperparameters and configuration setup
BATCH_SIZE = 32        # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 20        # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Data fractions for training and validation
train_data_fraction = 0.5  # Use 50% of the training data
valid_data_fraction = 0.5  # Use 50% of the validation data

# Device configuration: use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================================
# Directory Setup
# ================================================================
# Create directories to save results and figures
if not os.path.exists("../../reports/results"):
    os.makedirs("../../reports/results")

if not os.path.exists("../../reports/figures"):
    os.makedirs("../../reports/figures")

# Create a directory to save the models
if not os.path.exists("../../models"):
    os.makedirs("../../models")

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
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
    pin_memory=True, persistent_workers=True
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
    pin_memory=True, persistent_workers=True
)

# ================================================================
# Model Definitions
# ================================================================

# Get the number of classes
output_size = len(full_train_data.classes)

# ------------------------------
# Baseline Model Definition
# ------------------------------