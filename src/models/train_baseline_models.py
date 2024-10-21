# train_baseline_models.py

# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# tqdm for progress bars
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# For mixed precision training
from torch.cuda.amp import autocast, GradScaler

# For Weights & Biases integration
import wandb

# For model definitions
import timm

# ================================================================
# Helper Functions and Settings
# ================================================================
# Assuming helper_functions.py exists and contains set_seeds
# Adjust the path as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from helper_functions import set_seeds  # Adjust import based on your project structure

# ================================================================
# Setup Logging
# ================================================================
logging.basicConfig(
    filename='baseline_training_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# ================================================================
# Configuration and Settings
# ================================================================

# Set seeds for reproducibility
set_seeds(42)

# Hyperparameters
BATCH_SIZE = 64          # Adjust based on GPU memory
LEARNING_RATE = 1e-3     # Adjust as necessary
NUM_EPOCHS = 50          # Adjust based on experimentation
HEIGHT, WIDTH = 224, 224 # Image dimensions

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 10  # Increased patience for early stopping

# W&B Project Name
WANDB_PROJECT_NAME = "Plant_Leaf_Disease_Baselines"

# ================================================================
# Device Configuration
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs")
print(f"Using device: {device}")

# ================================================================
# Directory Setup
# ================================================================

# Define project root (assuming this script is in the project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Define directories for data and models
data_path = os.path.join(
    project_root,
    "data",
    "processed",
    "plant_leaf_disease_dataset",
    "single_task_disease",
)
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define output directories for results, figures, and models
output_dirs = [
    os.path.join(project_root, "reports", "results"),
    os.path.join(project_root, "reports", "figures"),
    os.path.join(project_root, "models", "baseline_models"),
]

# Create output directories if they don't exist
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Function to list directory contents
def list_directory_contents(directory, num_items=10):
    if os.path.exists(directory):
        contents = os.listdir(directory)
        print(
            f"Contents of {directory} ({len(contents)} items): {contents[:num_items]}..."
        )
    else:
        print(f"Directory does not exist: {directory}")

# Verify directories and list contents
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Validation directory exists: {os.path.exists(valid_dir)}")
list_directory_contents(train_dir, num_items=10)
list_directory_contents(valid_dir, num_items=10)

# ================================================================
# Load Label Mappings
# ================================================================

# Path to label mapping JSON
labels_mapping_path = os.path.join(data_path, "labels_mapping_single_task_disease.json")

# Load the label mapping
if os.path.exists(labels_mapping_path):
    with open(labels_mapping_path, "r") as f:
        labels_mapping = json.load(f)

    disease_to_idx = labels_mapping.get("disease_to_idx", {})
    if not disease_to_idx:
        print("Error: 'disease_to_idx' mapping not found in the JSON file.")
        sys.exit(1)

    idx_to_disease = {v: k for k, v in disease_to_idx.items()}
    print(f"Disease to Index Mapping: {disease_to_idx}")
    print(f"Index to Disease Mapping: {idx_to_disease}")
else:
    print(f"Warning: Label mapping file not found at {labels_mapping_path}. Exiting.")
    sys.exit(1)  # Exit, as proper label mapping is essential

# ================================================================
# Define Minority Classes
# ================================================================

# Define minority classes based on training label counts
# You can adjust the threshold as needed
minority_threshold = 1000  # Classes with fewer than 1000 samples are considered minority

# Path to training split CSV
train_split_csv = os.path.join(data_path, "train_split.csv")
if os.path.exists(train_split_csv):
    train_df = pd.read_csv(train_split_csv)
    train_label_counts = train_df['label'].value_counts().sort_index()
else:
    print(f"Error: Training split CSV not found at {train_split_csv}. Exiting.")
    sys.exit(1)

minority_classes = train_label_counts[train_label_counts < minority_threshold].index.tolist()

print(f"\nIdentified Minority Classes (count < {minority_threshold}):")
for cls in minority_classes:
    print(f"Class {cls} ({idx_to_disease.get(cls, 'Unknown')}) with {train_label_counts[cls]} samples")

# ================================================================
# Custom Dataset Class
# ================================================================

class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform_major=None, transform_minority=None,
                 minority_classes=None, image_col='image', label_col='label'):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            images_dir (str): Directory with all the images.
            transform_major (callable, optional): Transformations for majority classes.
            transform_minority (callable, optional): Transformations for minority classes.
            minority_classes (list, optional): List of minority class indices.
            image_col (str): Column name for image filenames in the CSV.
            label_col (str): Column name for labels in the CSV.
        """
        self.annotations = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform_major = transform_major
        self.transform_minority = transform_minority
        self.minority_classes = minority_classes if minority_classes else []
        self.image_col = image_col
        self.label_col = label_col

        # Verify required columns
        required_columns = [image_col, label_col]
        for col in required_columns:
            if col not in self.annotations.columns:
                raise ValueError(f"Missing required column '{col}' in CSV file.")

        # Ensure labels are integers
        if not pd.api.types.is_integer_dtype(self.annotations[self.label_col]):
            try:
                self.annotations[self.label_col] = self.annotations[self.label_col].astype(int)
                print(f"Converted labels in {csv_file} to integers.")
            except ValueError:
                print(f"Error: Labels in {csv_file} cannot be converted to integers.")
                self.annotations[self.label_col] = -1  # Assign invalid label

        # Debug: Print unique labels after conversion
        unique_labels = self.annotations[self.label_col].unique()
        print(f"Unique labels after conversion in {csv_file}: {unique_labels}")

        # Check labels are within [0, num_classes - 1]
        num_classes = len(disease_to_idx)
        valid_labels = self.annotations[self.label_col].between(0, num_classes - 1)
        invalid_count = len(self.annotations) - valid_labels.sum()
        if invalid_count > 0:
            print(f"Found {invalid_count} samples with invalid labels in {csv_file}. These will be skipped.")
            self.annotations = self.annotations[valid_labels].reset_index(drop=True)

        # Final count
        print(f"Number of samples after filtering in {csv_file}: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image filename and label
        img_name_full = self.annotations.iloc[idx][self.image_col]
        label_idx = self.annotations.iloc[idx][self.label_col]

        # Extract only the basename to avoid path duplication
        img_name = os.path.basename(img_name_full)

        # Full path to the image
        img_path = os.path.join(self.images_dir, img_name)

        # Open image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new("RGB", (HEIGHT, WIDTH), (0, 0, 0))

        # Apply class-specific transformations
        if label_idx in self.minority_classes and self.transform_minority:
            image = self.transform_minority(image)
        elif self.transform_major:
            image = self.transform_major(image)

        return image, label_idx

# ================================================================
# Data Transforms
# ================================================================

# Define transforms for majority classes
transform_major = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
        std=[0.229, 0.224, 0.225]     # Std for ImageNet
    ),
])

# Define transforms for minority classes with additional augmentations
transform_minority = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Additional flip
    transforms.RandomRotation(30),    # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
        std=[0.229, 0.224, 0.225]     # Std for ImageNet
    ),
])

# ================================================================
# Initialize Datasets and DataLoaders (Using WeightedRandomSampler)
# ================================================================

# Initialize training dataset
train_dataset = PlantDiseaseDataset(
    csv_file=train_split_csv,
    images_dir=train_dir,
    transform_major=transform_major,
    transform_minority=transform_minority,
    minority_classes=minority_classes,
    image_col='image',
    label_col='label'
)

# Path to validation split CSV
valid_split_csv = os.path.join(data_path, "valid_split.csv")
if os.path.exists(valid_split_csv):
    valid_df = pd.read_csv(valid_split_csv)
    valid_dataset = PlantDiseaseDataset(
        csv_file=valid_split_csv,
        images_dir=valid_dir,
        transform_major=transform_major,  # Validation should not have augmentation
        transform_minority=None,          # No augmentation for validation
        minority_classes=[],              # No augmentation needed
        image_col='image',
        label_col='label'
    )
else:
    print(f"Error: Validation split CSV not found at {valid_split_csv}. Exiting.")
    sys.exit(1)

# Create WeightedRandomSampler for the training DataLoader
# Compute class counts and weights
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1. / class_counts
samples_weight = class_weights[train_df['label'].values]
samples_weight = torch.from_numpy(samples_weight).double()

# Create the sampler
sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,  # Use sampler instead of shuffle
    num_workers=4, 
    pin_memory=True if torch.cuda.is_available() else False
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True if torch.cuda.is_available() else False
)

# Display dataset information
print(f"\nNumber of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of classes: {len(disease_to_idx)}")
print(f"Classes: {list(disease_to_idx.keys())}")

# Test fetching a single sample
if len(train_dataset) > 0:
    sample_image, sample_label = train_dataset[0]
    print(f"\nSample Image Shape: {sample_image.shape}")
    print(f"Sample Label Index: {sample_label}")
    print(f"Sample Label Name: {idx_to_disease.get(sample_label, 'Unknown')}")
else:
    print("\nTraining dataset is empty. Please check your dataset and label mappings.")

# ================================================================
# Baseline Model Definitions
# ================================================================

# ------------------------------
# Baseline Model Definition
# ------------------------------
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
        x = self.conv_block(x)  # Convolutional layers
        x = self.classifier(x)  # Fully connected layers
        return x

# ------------------------------
# ConvNetPlus Model Definition
# ------------------------------
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
        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # Second convolutional block
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
        # Fully connected layers
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

# ------------------------------
# TinyVGG Model Definition
# ------------------------------
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
        # First convolutional block
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
            nn.MaxPool2d(2),
        )
        # Second convolutional block
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
            nn.MaxPool2d(2),
        )
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 2 * (HEIGHT // 4) * (WIDTH // 4),
                out_features=output_size,
            ),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# ================================================================
# EfficientNetV2 Model Definition
# ================================================================

def get_efficientnetv2_model(output_size: int):
    """
    Instantiates an EfficientNetV2 model with pretrained weights.

    Args:
        output_size (int): Number of output classes.

    Returns:
        nn.Module: EfficientNetV2 model.
    """
    model = timm.create_model(
        "efficientnetv2_rw_s",  # Using EfficientNetV2 RW small version
        pretrained=True,         # Use pretrained weights
        num_classes=output_size, # Adjust output size to match number of classes
    )
    return model

# ================================================================
# Instantiate Models and Optimizers
# ================================================================

# Define the number of classes
output_size = len(disease_to_idx)

# Instantiate models
baseline_model = BaselineModel(input_shape=3, hidden_units=10, output_shape=output_size)
convnetplus_model = ConvNetPlus(input_shape=3, hidden_units=32, output_shape=output_size)
tinyvgg_model = TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size)
efficientnetv2_model = get_efficientnetv2_model(output_size=output_size)

# Move models to device
baseline_model = baseline_model.to(device)
convnetplus_model = convnetplus_model.to(device)
tinyvgg_model = tinyvgg_model.to(device)
efficientnetv2_model = efficientnetv2_model.to(device)

# If multiple GPUs are available, use DataParallel
if num_gpus > 1:
    baseline_model = nn.DataParallel(baseline_model)
    convnetplus_model = nn.DataParallel(convnetplus_model)
    tinyvgg_model = nn.DataParallel(tinyvgg_model)
    efficientnetv2_model = nn.DataParallel(efficientnetv2_model)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizers for each model
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
convnetplus_optimizer = optim.SGD(convnetplus_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
tinyvgg_optimizer = optim.RMSprop(tinyvgg_model.parameters(), lr=LEARNING_RATE)
efficientnetv2_optimizer = optim.Adam(
    efficientnetv2_model.parameters(), lr=LEARNING_RATE
)

# Learning rate schedulers
baseline_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    baseline_optimizer, mode="min", factor=0.1, patience=3, verbose=True
)
convnetplus_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    convnetplus_optimizer, mode="min", factor=0.1, patience=3, verbose=True
)
tinyvgg_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    tinyvgg_optimizer, mode="min", factor=0.1, patience=3, verbose=True
)
efficientnetv2_scheduler = optim.lr_scheduler.StepLR(
    efficientnetv2_optimizer, step_size=5, gamma=0.1, verbose=True
)

# ================================================================
# Callbacks for Training Monitoring
# ================================================================

# EarlyStopping and ModelCheckpoint will be handled within the training loop for each model

# ================================================================
# Training and Validation Functions
# ================================================================

from sklearn.metrics import classification_report, confusion_matrix

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler, epoch, model_name, log_interval=10):
    """
    Trains the model for one epoch using mixed precision.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to train on.
        scaler (GradScaler): GradScaler for mixed precision.
        epoch (int): Current epoch number.
        model_name (str): Name of the model for logging.
        log_interval (int): How often to log batch metrics.
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"{model_name} - Training", leave=False)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        # Enhanced Logging: Log every 'log_interval' batches
        if (batch_idx + 1) % log_interval == 0:
            unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
            class_distribution = dict(zip(unique, counts))
            wandb.log({
                f"{model_name}/train_loss": loss.item(),
                f"{model_name}/batch_train_accuracy": torch.sum(preds == labels.data).item() / inputs.size(0),
                f"{model_name}/batch_class_distribution": class_distribution
            })
            print(f"{model_name} - Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f} | Class Distribution: {class_distribution}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    wandb.log({
        f"{model_name}/epoch_train_loss": epoch_loss,
        f"{model_name}/epoch_train_accuracy": epoch_acc.item()
    })

    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, loss_fn, device, model_name, collect_metrics=True):
    """
    Validates the model.
    
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to validate on.
        model_name (str): Name of the model for logging.
        collect_metrics (bool): If True, collect labels and predictions.
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy, all_labels, all_preds)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{model_name} - Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Mixed precision inference
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            if collect_metrics:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    wandb.log({
        f"{model_name}/epoch_val_loss": epoch_loss,
        f"{model_name}/epoch_val_accuracy": epoch_acc.item()
    })

    return epoch_loss, epoch_acc.item(), all_labels, all_preds

# ================================================================
# Visualization Utilities
# ================================================================

def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies, model_name, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        train_losses (list): List of training losses.
        train_accuracies (list): List of training accuracies.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        wandb.log({f"{model_name}/training_validation_metrics": wandb.Image(save_path)})
        print(f"Training metrics plot saved at {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_model_post_training(model, dataloader, device, idx_to_disease, model_name, save_dir):
    """
    Evaluate the model on a dataset and print classification metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on.
        idx_to_disease (dict): Mapping from index to disease name.
        model_name (str): Name of the model for reporting.
        save_dir (str): Directory to save the confusion matrix plot.

    Returns:
        None
    """
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    total_samples = 0

    # Define loss function (same as training)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{model_name} - Evaluation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Mixed precision inference
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    print(f"\n{model_name} - Evaluation Loss: {avg_loss:.4f}")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=list(idx_to_disease.values()))
    print(f"\n{model_name} - Classification Report:")
    print(report)

    wandb.log({f"{model_name}/classification_report": report})

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(idx_to_disease.values()), 
                yticklabels=list(idx_to_disease.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    # Save the plot
    cm_save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_save_path)
    wandb.log({f"{model_name}/confusion_matrix": wandb.Image(cm_save_path)})
    plt.close()
    print(f"Confusion matrix saved at {cm_save_path}")

# ================================================================
# Initialize Weights & Biases (W&B)
# ================================================================

# Initialize W&B run
wandb.init(
    project=WANDB_PROJECT_NAME,
    name="Baseline_Models_Training",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "image_size": f"{HEIGHT}x{WIDTH}",
        "class_imbalance_handling": "WeightedRandomSampler",
    },
    save_code=True
)

# Get the run id for tracking
run_id = wandb.run.id
print(f"W&B Run ID: {run_id}")

# ================================================================
# Training Loop for Baseline Models
# ================================================================

def train_model(model, model_name, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, output_dirs, num_epochs=50):
    """
    Trains and validates a given model.

    Args:
        model (nn.Module): The model to train.
        model_name (str): Name of the model for identification.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        output_dirs (list): List of output directories for saving models and figures.
        num_epochs (int): Number of training epochs.

    Returns:
        None
    """
    # Initialize EarlyStopping and ModelCheckpoint
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, 
        verbose=True, 
        path=os.path.join(output_dirs[2], f"{model_name}_early_stop_model.pth")
    )
    model_checkpoint = ModelCheckpoint(
        path=os.path.join(output_dirs[2], f"{model_name}_best_val_loss_model.pth"), 
        verbose=True
    )

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Time tracking
    total_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n{model_name} - Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        epoch_start_time = time.time()
        
        # Training Phase
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            model_name=model_name
        )
        print(f"{model_name} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validation Phase
        val_loss, val_acc, _, _ = validate(
            model=model,
            dataloader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            model_name=model_name
        )
        print(f"{model_name} - Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}%")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"{model_name} - Epoch Duration: {epoch_duration:.2f} seconds")
        
        # Append metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduler step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = scheduler.get_last_lr()[0]
        wandb.log({f"{model_name}/learning_rate": current_lr})
        
        # Model checkpoint based on validation loss
        model_checkpoint(val_loss, model)
        
        # Early Stopping based on validation loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"{model_name} - Early stopping triggered!")
            break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n{model_name} - Total Training Time: {total_duration/60:.2f} minutes")

    # Log total training time to W&B
    wandb.log({f"{model_name}/total_training_time_minutes": total_duration/60})

    # Plot training metrics
    plot_save_path = os.path.join(output_dirs[1], f"{model_name}_training_validation_metrics.png")
    plot_training_metrics(
        train_losses, 
        train_accuracies, 
        val_losses, 
        val_accuracies, 
        model_name=model_name,
        save_path=plot_save_path
    )

    # Perform post-training evaluation on the validation set
    evaluate_model_post_training(
        model=model, 
        dataloader=valid_loader, 
        device=device, 
        idx_to_disease=idx_to_disease, 
        model_name=model_name, 
        save_dir=output_dirs[1]
    )

# ================================================================
# Main Execution
# ================================================================

if __name__ == "__main__":
    # List of models to train
    models = [
        {
            "model": baseline_model,
            "name": "BaselineModel",
            "optimizer": baseline_optimizer,
            "scheduler": baseline_scheduler
        },
        {
            "model": convnetplus_model,
            "name": "ConvNetPlus",
            "optimizer": convnetplus_optimizer,
            "scheduler": convnetplus_scheduler
        },
        {
            "model": tinyvgg_model,
            "name": "TinyVGG",
            "optimizer": tinyvgg_optimizer,
            "scheduler": tinyvgg_scheduler
        },
        {
            "model": efficientnetv2_model,
            "name": "EfficientNetV2",
            "optimizer": efficientnetv2_optimizer,
            "scheduler": efficientnetv2_scheduler
        },
    ]

    for m in models:
        print(f"\n{'='*50}\nStarting Training for {m['name']}\n{'='*50}")
        wandb.run.name = f"{m['name']}_Training"
        wandb.run.save()
        train_model(
            model=m["model"],
            model_name=m["name"],
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            optimizer=m["optimizer"],
            scheduler=m["scheduler"],
            device=device,
            output_dirs=output_dirs,
            num_epochs=NUM_EPOCHS
        )

    # Finalize W&B run
    wandb.finish()
