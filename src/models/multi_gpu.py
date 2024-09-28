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
from sklearn.metrics import confusion_matrix, accuracy_score

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds, accuracy_fn  # Custom helper functions
from helper_functions import *  # Import helper functions and matplotlib settings

# ================================================================
# Configuration and Settings
# ================================================================
# Set seeds for reproducibility
set_seeds(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32        # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 10        # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Data fractions for training and validation
train_data_fraction = 0.5  # Use 50% of the training data
valid_data_fraction = 0.1  # Use 10% of the validation data

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
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * (HEIGHT // 2) * (WIDTH // 2),
                      out_features=output_shape),
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
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # Fully connected layers
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
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 2 * (HEIGHT // 4) * (WIDTH // 4),
                      out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# ================================================================
# Instantiate Models and Optimizers
# ================================================================

# Get the number of classes
output_size = len(full_train_data.classes)

# Instantiate the models
baseline_model = BaselineModel(input_shape=3, hidden_units=8, output_shape=output_size)
convnetplus_model = ConvNetPlus(input_shape=3, hidden_units=16, output_shape=output_size)
tinyvgg_model = TinyVGG(input_shape=3, hidden_units=64, output_shape=output_size)

# Move models to device and wrap with DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    baseline_model = nn.DataParallel(baseline_model)
    convnetplus_model = nn.DataParallel(convnetplus_model)
    tinyvgg_model = nn.DataParallel(tinyvgg_model)

baseline_model.to(device)
convnetplus_model.to(device)
tinyvgg_model.to(device)

# Define loss function and optimizers
loss_fn = nn.CrossEntropyLoss()

# Optimizers for each model
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
convnetplus_optimizer = optim.SGD(convnetplus_model.parameters(), lr=LEARNING_RATE)
tinyvgg_optimizer = optim.RMSprop(tinyvgg_model.parameters(), lr=LEARNING_RATE)

# ================================================================
# Training and Evaluation Functions
# ================================================================

# ------------------------------
# Training Function
# ------------------------------
def train_model(model, dataloader, optimizer, loss_fn, accuracy_fn, epoch):
    """
    Trains the model for one epoch and returns the average training loss and accuracy.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (nn.Module): Loss function.
        accuracy_fn (function): Function to compute accuracy.
        epoch (int): Current epoch number.

    Returns:
        tuple: Average training loss and accuracy.
    """
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False)):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    avg_train_loss = train_loss / len(dataloader)
    avg_train_acc = train_acc / len(dataloader)
    return avg_train_loss, avg_train_acc

# ------------------------------
# Evaluation Function
# ------------------------------
def eval_model(model, dataloader, loss_fn, accuracy_fn):
    """
    Evaluates the model on the given data loader.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for validation data.
        loss_fn (nn.Module): Loss function.
        accuracy_fn (function): Function to compute accuracy.

    Returns:
        dict: Dictionary with validation loss and accuracy.
    """
    model.eval()
    valid_loss, valid_acc = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            valid_loss += loss_fn(y_pred, y).item()
            valid_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    avg_valid_loss = valid_loss / len(dataloader)
    avg_valid_acc = valid_acc / len(dataloader)
    return {"model_loss": avg_valid_loss, "model_acc": avg_valid_acc}

# ------------------------------
# Training and Evaluation Loop
# ------------------------------
def train_and_evaluate(
    model, train_loader, valid_loader, optimizer, loss_fn, accuracy_fn,
    num_epochs, model_name, optimizer_name, early_stopping_patience=1
):
    """
    Trains and evaluates the model for a specified number of epochs with early stopping.

    Args:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        optimizer (optim.Optimizer): Optimizer for training the model.
        loss_fn (nn.Module): Loss function.
        accuracy_fn (function): Function to compute accuracy.
        num_epochs (int): Maximum number of epochs for training.
        model_name (str): Name of the model (for logging purposes).
        optimizer_name (str): Name of the optimizer used.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        tuple: (list of epoch results, best result dictionary)
    """
    results = []
    total_start_time = time.time()

    best_valid_acc = 0
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    # Lists to store metrics for plotting
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    epoch_durations = []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        avg_train_loss, avg_train_acc = train_model(
            model, train_loader, optimizer, loss_fn, accuracy_fn, epoch
        )
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.2f}%")

        # Validation
        valid_results = eval_model(model, valid_loader, loss_fn, accuracy_fn)
        print(
            f"Validation Loss: {valid_results['model_loss']:.4f}, "
            f"Validation Accuracy: {valid_results['model_acc']:.2f}%"
        )

        # Epoch duration
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

        # Append metrics for plotting
        train_losses.append(avg_train_loss)
        valid_losses.append(valid_results["model_loss"])
        train_accuracies.append(avg_train_acc)
        valid_accuracies.append(valid_results["model_acc"])
        epoch_durations.append(epoch_duration)

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
                f"Early stopping triggered after {early_stopping_patience} "
                f"epoch(s) with no improvement."
            )
            break

    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Save the best model
    torch.save(best_model_state, f"../../models/{model_name}_best_model.pth")

    # Save training history for plotting
    history = {
        "epochs": list(range(1, len(train_losses) + 1)),
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_accuracies": train_accuracies,
        "valid_accuracies": valid_accuracies,
        "epoch_durations": epoch_durations,
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
        "best_valid_loss": results[best_epoch - 1]["valid_loss"],
        "best_valid_acc": best_valid_acc,
        "total_training_time": total_training_time,
    }
    return results, best_result

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
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_losses'], label='Training Loss')
    plt.plot(history['epochs'], history['valid_losses'], label='Validation Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
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
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_accuracies'], label='Training Accuracy')
    plt.plot(history['epochs'], history['valid_accuracies'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
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
    early_stopping_patience=2,
)

# ------------------------------
# Training ConvNetPlus Model
# ------------------------------
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
    early_stopping_patience=2,
)

# ------------------------------
# Training TinyVGG Model
# ------------------------------
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
    early_stopping_patience=2,
)

# ================================================================
# Results Compilation and Saving
# ================================================================

# Combine the best results from each model
best_results_df = pd.DataFrame([baseline_best, convnetplus_best, tinyvgg_best])

# Display the best results DataFrame for comparison
print("\nBest Results for Each Model:")
print(best_results_df)

# Save the combined best results to a CSV file
best_results_df.to_csv("../../reports/results/model_best_results.csv", index=False)

# ================================================================
# Comparison Plots Across Models
# ================================================================

def plot_model_comparison(results_list, metric, title, ylabel, filename):
    """
    Plots comparison of a specific metric across different models.

    Args:
        results_list (list): List of result dictionaries for each model.
        metric (str): Metric to compare ('train_loss', 'valid_loss', 'train_acc', 'valid_acc').
        title (str): Title of the plot.
        ylabel (str): Y-axis label.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for result in results_list:
        epochs = [epoch['epoch'] for epoch in result['results']]
        values = [epoch[metric] for epoch in result['results']]
        plt.plot(epochs, values, label=result['model_name'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{filename}.pdf")
    plt.close()

# Prepare results for plotting
baseline_plot_data = {'results': baseline_results, 'model_name': 'BaselineModel'}
convnetplus_plot_data = {'results': convnetplus_results, 'model_name': 'ConvNetPlus'}
tinyvgg_plot_data = {'results': tinyvgg_results, 'model_name': 'TinyVGG'}

all_results = [baseline_plot_data, convnetplus_plot_data, tinyvgg_plot_data]

# Plot total training time comparison
model_names = [baseline_best['model'], convnetplus_best['model'], tinyvgg_best['model']]
training_times = [
    baseline_best['total_training_time'],
    convnetplus_best['total_training_time'],
    tinyvgg_best['total_training_time']
]

plt.figure(figsize=(10, 6))
plt.bar(model_names, training_times, color=['blue', 'orange', 'green'])
plt.title('Total Training Time Comparison')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.savefig("../../reports/figures/training_time_comparison.pdf")
plt.close()



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
    model_names = [result['model'] for result in results_list]
    metric_values = [result[metric] for result in results_list]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, metric_values, color=['blue', 'orange', 'green'])
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f"../../reports/figures/{filename}.pdf")
    plt.close()

# ================================================================
# Generate Bar Charts for Accuracy and Loss Comparison
# ================================================================

# Plot bar chart for best validation accuracy comparison
plot_bar_comparison(
    results_list=[baseline_best, convnetplus_best, tinyvgg_best],
    metric='best_valid_acc',
    ylabel='Best Validation Accuracy (%)',
    title='Best Validation Accuracy Comparison',
    filename='best_validation_accuracy_comparison'
)

# Plot bar chart for best validation loss comparison
plot_bar_comparison(
    results_list=[baseline_best, convnetplus_best, tinyvgg_best],
    metric='best_valid_loss',
    ylabel='Best Validation Loss',
    title='Best Validation Loss Comparison',
    filename='best_validation_loss_comparison'
)

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

    plt.figure(figsize=(20, 15))
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

                prob = probs[idx][preds[idx]].item() * 100  # Get the probability of the predicted class
                true_label = class_names[labels[idx]]
                pred_label = class_names[preds[idx]]
                title_color = 'green' if preds[idx] == labels[idx] else 'red'

                plt.subplot(1, num_samples, images_displayed)
                plt.imshow(image)
                plt.title(f'True: {true_label}\nPred: {pred_label}\nConfidence: {prob:.1f}%',
                          color=title_color, fontsize=8)
                plt.axis('off')

            if images_displayed >= num_samples:
                break

    plt.suptitle(f'Sample Predictions by {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"../../reports/figures/{model_name}_sample_predictions.pdf")
    plt.close()

# Plot sample predictions for each model
plot_sample_predictions(baseline_model, valid_loader, 'BaselineModel')
plot_sample_predictions(convnetplus_model, valid_loader, 'ConvNetPlus')
plot_sample_predictions(tinyvgg_model, valid_loader, 'TinyVGG')

# ================================================================
# Confusion Matrix Generation
# ================================================================

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
        for images, labels in tqdm(data_loader, desc=f"Generating Confusion Matrix for {model_name}", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(30, 20))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_confusion_matrix.pdf")
    plt.close()

# Generate confusion matrices for each model
generate_confusion_matrix(baseline_model, valid_loader, 'BaselineModel')
generate_confusion_matrix(convnetplus_model, valid_loader, 'ConvNetPlus')
generate_confusion_matrix(tinyvgg_model, valid_loader, 'TinyVGG')
