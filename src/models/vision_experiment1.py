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
BATCH_SIZE = 32  # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 50  # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Data fractions for training and validation
train_data_fraction = 1.0  # Use 100% of the training data
valid_data_fraction = 1.0  # Use 100% of the validation data

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
np.random.seed(42)  # Ensure reproducibility
train_indices = np.random.choice(len(full_train_data), train_size, replace=False)
valid_indices = np.random.choice(len(full_valid_data), valid_size, replace=False)

train_data = Subset(full_train_data, train_indices)
valid_data = Subset(full_valid_data, valid_indices)

# Data Loaders for training and validation data
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

# ================================================================
# Model Definitions
# ================================================================

# Get the number of classes
output_size = len(full_train_data.classes)


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


# ------------------------------
# EfficientNetV2 Model Definition
# ------------------------------
# EfficientNetV2 Model Definition
# Instantiate the EfficientNetV2 model with pretrained weights
efficientnetv2_model = timm.create_model(
    "efficientnetv2_rw_s",  # Using EfficientNetV2 RW small version
    pretrained=True,  # Use pretrained weights
    num_classes=output_size,  # Adjust output size to match number of classes
)


# ================================================================
# Instantiate Models and Optimizers
# ================================================================
# Move models to device and wrap with DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    baseline_model = nn.DataParallel(
        BaselineModel(input_shape=3, hidden_units=10, output_shape=output_size)
    )
    convnetplus_model = nn.DataParallel(
        ConvNetPlus(input_shape=3, hidden_units=32, output_shape=output_size)
    )
    tinyvgg_model = nn.DataParallel(
        TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size)
    )
    efficientnetv2_model = nn.DataParallel(efficientnetv2_model)
else:
    baseline_model = BaselineModel(
        input_shape=3, hidden_units=10, output_shape=output_size
    )
    convnetplus_model = ConvNetPlus(
        input_shape=3, hidden_units=32, output_shape=output_size
    )
    tinyvgg_model = TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size)

baseline_model.to(device)
convnetplus_model.to(device)
tinyvgg_model.to(device)
efficientnetv2_model.to(device)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizers for each model
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
convnetplus_optimizer = optim.SGD(convnetplus_model.parameters(), lr=LEARNING_RATE)
tinyvgg_optimizer = optim.RMSprop(tinyvgg_model.parameters(), lr=LEARNING_RATE)
efficientnetv2_optimizer = optim.Adam(
    efficientnetv2_model.parameters(), lr=LEARNING_RATE
)

# Learning rate schedulers
baseline_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    baseline_optimizer, mode="min", factor=0.1, patience=3
)
convnetplus_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    convnetplus_optimizer, mode="min", factor=0.1, patience=3
)
tinyvgg_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    tinyvgg_optimizer, mode="min", factor=0.1, patience=3
)
efficientnetv2_scheduler = optim.lr_scheduler.StepLR(
    efficientnetv2_optimizer, step_size=5, gamma=0.1
)


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
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs,
    model_name,
    optimizer_name,
    early_stopping_patience=5,
):
    """
    Training loop for the model with early stopping.

    Args:
        model (nn.Module): The model to train.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (optim.lr_scheduler): The learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        device (torch.device): The device to train on.
        num_epochs (int): Number of epochs to train.
        model_name (str): Name of the model (for saving).
        optimizer_name (str): Name of the optimizer.
        early_stopping_patience (int): Patience for early stopping.

    Returns:
        tuple: (trained model, best result dictionary)
    """
    # Initialize variables
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0)
    best_valid_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}", leave=False
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validation", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

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

        # Step the scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Early Stopping
        early_stopping(avg_val_loss)
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"../../models/{model_name}_best_model.pth")
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    total_training_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

    # Load the best model state
    model.load_state_dict(torch.load(f"../../models/{model_name}_best_model.pth"))

    # Save training history for plotting
    history = {
        "epochs": list(range(1, len(train_losses) + 1)),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # Plot and save training and validation loss
    plot_loss(history, model_name)

    # Plot and save training and validation accuracy
    plot_accuracy(history, model_name)

    # Return best results
    best_result = {
        "model": model_name,
        "optimizer": optimizer_name,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_valid_acc": val_accuracies[best_epoch - 1],
        "total_training_time": total_training_time,
    }

    return model, best_result


# ================================================================
# Plotting Functions
# ================================================================
def plot_loss(history, model_name):
    """
    Plots training and validation loss over epochs.

    Args:
        history (dict): Dictionary containing loss and accuracy history.
        model_name (str): Name of the model (for saving the plot).
    """
    plt.figure()
    plt.plot(
        history["epochs"], history["train_losses"], label="Training Loss", linewidth=2
    )
    plt.plot(
        history["epochs"], history["val_losses"], label="Validation Loss", linewidth=2
    )
    plt.title(f"{model_name} Loss over Epochs", fontsize=18)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_loss.pdf")
    plt.close()


def plot_accuracy(history, model_name):
    """
    Plots training and validation accuracy over epochs.

    Args:
        history (dict): Dictionary containing loss and accuracy history.
        model_name (str): Name of the model (for saving the plot).
    """
    plt.figure()
    plt.plot(
        history["epochs"],
        history["train_accuracies"],
        label="Training Accuracy",
        linewidth=2,
    )
    plt.plot(
        history["epochs"],
        history["val_accuracies"],
        label="Validation Accuracy",
        linewidth=2,
    )
    plt.title(f"{model_name} Accuracy over Epochs", fontsize=18)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_accuracy.pdf")
    plt.close()


# ================================================================
# Training the Models
# ================================================================

# ------------------------------
# Training Baseline Model
# ------------------------------
print("\nTraining Baseline Model")
baseline_model, baseline_best = train_model(
    model=baseline_model,
    optimizer=baseline_optimizer,
    scheduler=baseline_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="BaselineModel",
    optimizer_name="Adam",
    early_stopping_patience=3,
)

# ------------------------------
# Training ConvNetPlus Model
# ------------------------------
print("\nTraining ConvNetPlus Model")
convnetplus_model, convnetplus_best = train_model(
    model=convnetplus_model,
    optimizer=convnetplus_optimizer,
    scheduler=convnetplus_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="ConvNetPlus",
    optimizer_name="SGD",
    early_stopping_patience=3,
)

# ------------------------------
# Training TinyVGG Model
# ------------------------------
print("\nTraining TinyVGG Model")
tinyvgg_model, tinyvgg_best = train_model(
    model=tinyvgg_model,
    optimizer=tinyvgg_optimizer,
    scheduler=tinyvgg_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="TinyVGG",
    optimizer_name="RMSprop",
    early_stopping_patience=3,
)

# ------------------------------
# Training EfficientNetV2 Model
# ------------------------------
print("\nTraining EfficientNetV2 Model")
efficientnetv2_model, efficientnetv2_best = train_model(
    model=efficientnetv2_model,
    optimizer=efficientnetv2_optimizer,
    scheduler=efficientnetv2_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="EfficientNetV2",
    optimizer_name="Adam",
    early_stopping_patience=3,
)

# ================================================================
# Results Compilation and Saving
# ================================================================
# Combine the best results from each model
best_results_df = pd.DataFrame(
    [baseline_best, convnetplus_best, tinyvgg_best, efficientnetv2_best]
)

# Display the best results DataFrame for comparison
print("\nBest Results for Each Model:")
print(best_results_df)

# Save the combined best results to a CSV file
best_results_df.to_csv("../../reports/results/model_best_results.csv", index=False)


# ================================================================
# Bar Chart Comparison Function
# ================================================================
def plot_bar_comparison(results_list, metric, ylabel, title, filename):
    """
    Plots a bar chart comparing the specified metric (accuracy or loss) for each model.

    Args:
        results_list (list): List of result dictionaries for each model.
        metric (str): Metric to compare ('best_valid_acc', 'best_valid_loss').
        ylabel (str): Label for the Y-axis (e.g., 'Accuracy', 'Loss').
        title (str): Title of the bar chart.
        filename (str): Filename to save the chart as a PDF.
    """
    model_names = [result["model"] for result in results_list]
    metric_values = [result[metric] for result in results_list]

    plt.figure()
    plt.bar(
        model_names, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Model", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f"../../reports/figures/{filename}.pdf")
    plt.close()


# ================================================================
# Generate Bar Charts for Accuracy and Loss Comparison
# ================================================================
# Plot bar chart for best validation accuracy comparison
plot_bar_comparison(
    results_list=[baseline_best, convnetplus_best, tinyvgg_best, efficientnetv2_best],
    metric="best_valid_acc",
    ylabel="Best Validation Accuracy (%)",
    title="Best Validation Accuracy Comparison",
    filename="best_validation_accuracy_comparison",
)

# Plot bar chart for best validation loss comparison
plot_bar_comparison(
    results_list=[baseline_best, convnetplus_best, tinyvgg_best, efficientnetv2_best],
    metric="best_valid_loss",
    ylabel="Best Validation Loss",
    title="Best Validation Loss Comparison",
    filename="best_validation_loss_comparison",
)

# Plot total training time comparison
model_names = [
    baseline_best["model"],
    convnetplus_best["model"],
    tinyvgg_best["model"],
    efficientnetv2_best["model"],
]
training_times = [
    baseline_best["total_training_time"],
    convnetplus_best["total_training_time"],
    tinyvgg_best["total_training_time"],
    efficientnetv2_best["total_training_time"],
]

plt.figure()
plt.bar(model_names, training_times, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
plt.title("Total Training Time Comparison", fontsize=18)
plt.xlabel("Model", fontsize=16)
plt.ylabel("Time (seconds)", fontsize=16)
plt.tight_layout()
plt.savefig("../../reports/figures/training_time_comparison.pdf")
plt.close()


# ================================================================
# Additional Visualizations
# ================================================================
# Function to plot sample images with predictions and probabilities
def plot_sample_predictions(model, data_loader, model_name, num_samples=5):
    """
    Plots sample images with true labels, model predictions, and prediction probabilities.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the data to sample from.
        model_name (str): Name of the model (for saving the plot).
        num_samples (int): Number of samples to display.
    """
    model.eval()
    class_names = full_train_data.classes
    images_displayed = 0

    plt.figure(figsize=(20, 5))
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            for idx in range(images.size(0)):
                if images_displayed >= num_samples:
                    break
                images_displayed += 1
                image = images[idx].cpu().numpy().transpose((1, 2, 0))
                image = np.clip(image, 0, 1)

                prob = (
                    probs[idx][preds[idx]].item() * 100
                )  # Get the probability of the predicted class
                true_label = class_names[labels[idx]]
                pred_label = class_names[preds[idx]]
                title_color = "green" if preds[idx] == labels[idx] else "red"

                plt.subplot(1, num_samples, images_displayed)
                plt.imshow(image)
                plt.title(
                    f"True: {true_label}\nPred: {pred_label}\nConf: {prob:.1f}%",
                    color=title_color,
                    fontsize=10,
                )
                plt.axis("off")

            if images_displayed >= num_samples:
                break

    plt.suptitle(f"Sample Predictions by {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"../../reports/figures/{model_name}_sample_predictions.pdf")
    plt.close()


# Function to generate confusion matrix
def generate_confusion_matrix(model, data_loader, model_name):
    """
    Generates and saves a confusion matrix for the model's predictions on the validation set.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the validation data.
        model_name (str): Name of the model (for saving the plot).
    """
    model.eval()
    all_preds = []
    all_labels = []
    class_names = full_train_data.classes

    with torch.no_grad():
        for images, labels in tqdm(
            data_loader,
            desc=f"Generating Confusion Matrix for {model_name}",
            leave=False,
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title(f"Confusion Matrix for {model_name}", fontsize=18)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_confusion_matrix.pdf")
    plt.close()


# Generate sample predictions and confusion matrices for each model
models = [
    (baseline_model, "BaselineModel"),
    (convnetplus_model, "ConvNetPlus"),
    (tinyvgg_model, "TinyVGG"),
    (efficientnetv2_model, "EfficientNetV2"),
]

for model, model_name in models:
    print(f"\nGenerating visualizations for {model_name}")
    plot_sample_predictions(model, valid_loader, model_name)
    generate_confusion_matrix(model, valid_loader, model_name)
