# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import time
import random
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

# ================================================================
# Configuration and Settings
# ================================================================


# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seeds(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
HEIGHT, WIDTH = 224, 224

# Data fractions for training and validation
train_data_fraction = 1.0  # Use 100% of the training data
valid_data_fraction = 1.0  # Use 100% of the validation data

# Device configuration: use multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
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

# Define image transformations
# For training data, include data augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]
)

# For validation data, use deterministic transforms
valid_transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
    ]
)
# num_workers = os.cpu_count() if torch.cuda.is_available() else 0

# Load the full training and validation datasets
full_train_data = datasets.ImageFolder(train_dir, transform=train_transform)
full_valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)

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
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
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
    """

    def __init__(self, input_shape, hidden_units, output_shape):
        super(BaselineModel, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * (HEIGHT // 2) * (WIDTH // 2),
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


# ------------------------------
# ConvNetPlus Model Definition
# ------------------------------
class ConvNetPlus(nn.Module):
    """
    Defines an improved model with additional layers, batch normalization, and dropout.
    """

    def __init__(self, input_shape, hidden_units, output_shape):
        super(ConvNetPlus, self).__init__()
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
    """

    def __init__(self, input_shape, hidden_units, output_shape):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
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
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units * 2,
                out_channels=hidden_units * 2,
                kernel_size=3,
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


# ================================================================
# Instantiate Models and Optimizers
# ================================================================

# Initialize models
baseline_model = BaselineModel(
    input_shape=3, hidden_units=10, output_shape=output_size
).to(device)
convnetplus_model = ConvNetPlus(
    input_shape=3, hidden_units=32, output_shape=output_size
).to(device)
tinyvgg_model = TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size).to(
    device
)

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    baseline_model = nn.DataParallel(baseline_model)
    convnetplus_model = nn.DataParallel(convnetplus_model)
    tinyvgg_model = nn.DataParallel(tinyvgg_model)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizers
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
convnetplus_optimizer = optim.Adam(convnetplus_model.parameters(), lr=LEARNING_RATE)
tinyvgg_optimizer = optim.Adam(tinyvgg_model.parameters(), lr=LEARNING_RATE)

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

# ================================================================
# EarlyStopping Class Definition
# ================================================================


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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
                print(
                    f"Early stopping triggered after {self.patience} epochs with no improvement."
                )
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


# ================================================================
# Training and Evaluation Functions
# ================================================================


def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_preds / total_samples * 100

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_preds / total_samples * 100

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs,
    model_name,
    early_stopping_patience=5,
):
    # Initialize history
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    early_stopping = EarlyStopping(patience=early_stopping_patience)
    best_valid_loss = float("inf")
    best_model_wts = None

    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        valid_loss, valid_acc = validate_one_epoch(model, valid_loader, device)

        # Step the scheduler
        scheduler.step(valid_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")

        # Early Stopping
        early_stopping(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, f"../../models/{model_name}_best.pth")

        if early_stopping.early_stop:
            break

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Load best model weights
    model.load_state_dict(torch.load(f"../../models/{model_name}_best.pth"))

    # Return model and history
    return model, history, best_epoch, best_valid_loss, total_time


# ================================================================
# Plotting Functions
# ================================================================


def plot_metrics(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../../reports/figures/{model_name}_loss.pdf")
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["valid_acc"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../../reports/figures/{model_name}_accuracy.pdf")
    plt.close()


# ================================================================
# Training the Models
# ================================================================

# Training Baseline Model
print("\nTraining Baseline Model")
(
    baseline_model,
    baseline_history,
    baseline_best_epoch,
    baseline_best_valid_loss,
    baseline_total_time,
) = train_model(
    model=baseline_model,
    optimizer=baseline_optimizer,
    scheduler=baseline_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="BaselineModel",
)

plot_metrics(baseline_history, "BaselineModel")

# Training ConvNetPlus Model
print("\nTraining ConvNetPlus Model")
(
    convnetplus_model,
    convnetplus_history,
    convnetplus_best_epoch,
    convnetplus_best_valid_loss,
    convnetplus_total_time,
) = train_model(
    model=convnetplus_model,
    optimizer=convnetplus_optimizer,
    scheduler=convnetplus_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="ConvNetPlus",
)

plot_metrics(convnetplus_history, "ConvNetPlus")

# Training TinyVGG Model
print("\nTraining TinyVGG Model")
(
    tinyvgg_model,
    tinyvgg_history,
    tinyvgg_best_epoch,
    tinyvgg_best_valid_loss,
    tinyvgg_total_time,
) = train_model(
    model=tinyvgg_model,
    optimizer=tinyvgg_optimizer,
    scheduler=tinyvgg_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name="TinyVGG",
)

plot_metrics(tinyvgg_history, "TinyVGG")

# ================================================================
# Results Compilation and Saving
# ================================================================

# Prepare best results
best_results = [
    {
        "model": "BaselineModel",
        "best_epoch": baseline_best_epoch,
        "best_valid_loss": baseline_best_valid_loss,
        "total_training_time": baseline_total_time,
        "best_valid_acc": baseline_history["valid_acc"][baseline_best_epoch - 1],
    },
    {
        "model": "ConvNetPlus",
        "best_epoch": convnetplus_best_epoch,
        "best_valid_loss": convnetplus_best_valid_loss,
        "total_training_time": convnetplus_total_time,
        "best_valid_acc": convnetplus_history["valid_acc"][convnetplus_best_epoch - 1],
    },
    {
        "model": "TinyVGG",
        "best_epoch": tinyvgg_best_epoch,
        "best_valid_loss": tinyvgg_best_valid_loss,
        "total_training_time": tinyvgg_total_time,
        "best_valid_acc": tinyvgg_history["valid_acc"][tinyvgg_best_epoch - 1],
    },
]

best_results_df = pd.DataFrame(best_results)

# Display the best results DataFrame for comparison
print("\nBest Results for Each Model:")
print(best_results_df)

# Save the combined best results to a CSV file
best_results_df.to_csv("../../reports/results/model_best_results.csv", index=False)

# ================================================================
# Bar Chart Comparison Function
# ================================================================


def plot_bar_comparison(results_df, metric, ylabel, title, filename):
    """
    Plots a bar chart comparing the specified metric (accuracy or loss) for each model.
    """
    model_names = results_df["model"]
    metric_values = results_df[metric]

    plt.figure()
    plt.bar(model_names, metric_values)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{filename}.pdf")
    plt.close()


# ================================================================
# Generate Bar Charts for Accuracy and Loss Comparison
# ================================================================

# Plot bar chart for best validation accuracy comparison
plot_bar_comparison(
    results_df=best_results_df,
    metric="best_valid_acc",
    ylabel="Best Validation Accuracy (%)",
    title="Best Validation Accuracy Comparison",
    filename="best_validation_accuracy_comparison",
)

# Plot bar chart for best validation loss comparison
plot_bar_comparison(
    results_df=best_results_df,
    metric="best_valid_loss",
    ylabel="Best Validation Loss",
    title="Best Validation Loss Comparison",
    filename="best_validation_loss_comparison",
)

# Plot total training time comparison
plot_bar_comparison(
    results_df=best_results_df,
    metric="total_training_time",
    ylabel="Total Training Time (s)",
    title="Total Training Time Comparison",
    filename="training_time_comparison",
)

# ================================================================
# Additional Visualizations
# ================================================================


def plot_sample_predictions(model, data_loader, model_name, num_samples=5):
    model.eval()
    class_names = full_train_data.classes
    images_displayed = 0

    plt.figure(figsize=(15, 3))
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

                prob = probs[idx][preds[idx]].item() * 100
                true_label = class_names[labels[idx]]
                pred_label = class_names[preds[idx]]
                title_color = "green" if preds[idx] == labels[idx] else "red"

                plt.subplot(1, num_samples, images_displayed)
                plt.imshow(image)
                plt.title(
                    f"True: {true_label}\nPred: {pred_label}\nConf: {prob:.1f}%",
                    color=title_color,
                    fontsize=8,
                )
                plt.axis("off")

            if images_displayed >= num_samples:
                break

    plt.suptitle(f"Sample Predictions by {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_sample_predictions.pdf")
    plt.close()


def generate_confusion_matrix(model, data_loader, model_name):
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
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_confusion_matrix.pdf")
    plt.close()


# Generate sample predictions and confusion matrices for each model
models = [
    (baseline_model, "BaselineModel"),
    (convnetplus_model, "ConvNetPlus"),
    (tinyvgg_model, "TinyVGG"),
]

for model, model_name in models:
    print(f"\nGenerating visualizations for {model_name}")
    plot_sample_predictions(model, valid_loader, model_name)
    generate_confusion_matrix(model, valid_loader, model_name)
