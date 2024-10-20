"""
A series of helper functions used for machine learning models, including data downloading, 
visualization, metrics, and setting random seeds.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from torch import nn
from pathlib import Path
from typing import List, Tuple
import torchvision
import requests
from cycler import cycler  # Correct import for cycler


# Helper function to walk through a directory and count files in subdirectories
def walk_through_dir(dir_path: str) -> None:
    """
    Walks through a directory and prints the number of images (files) and subdirectories.

    Args:
    dir_path (str): Path to directory to walk through.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# Plotting decision boundaries for 2D data (useful for binary classification)
def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    """
    Plots decision boundaries for a model's predictions.

    Args:
    model (torch.nn.Module): Trained model.
    X (torch.Tensor): Input features (2D data).
    y (torch.Tensor): True labels.
    """
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    model.eval()
    with torch.no_grad():
        y_logits = model(X_to_pred_on)

    y_pred = (
        torch.softmax(y_logits, dim=1).argmax(dim=1)
        if len(torch.unique(y)) > 2
        else torch.round(torch.sigmoid(y_logits))
    )
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Helper function to plot loss and accuracy curves
def plot_loss_curves(results: dict) -> None:
    """
    Plots loss and accuracy curves given training results.

    Args:
    results (dict): A dictionary with 'train_loss', 'train_acc', 'test_loss', 'test_acc' as keys.
    """
    loss, test_loss = results["train_loss"], results["test_loss"]
    acc, test_acc = results["train_acc"], results["test_acc"]
    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Function to plot predictions vs. ground truth
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
) -> None:
    """
    Plots the training data, test data, and predictions if provided.

    Args:
    train_data, train_labels, test_data, test_labels (torch.Tensor): Data for plotting.
    predictions (torch.Tensor, optional): Predictions to compare against test labels.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})


# Calculate classification accuracy
def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates accuracy between true labels and predicted labels.

    Args:
    y_true (torch.Tensor): Ground truth labels.
    y_pred (torch.Tensor): Model predictions.

    Returns:
    float: Accuracy value.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


# Timing helper function to measure training time
def print_train_time(start: float, end: float, device=None) -> float:
    """
    Prints and returns the time taken for training.

    Args:
    start (float): Start time.
    end (float): End time.
    device (optional): Device used for training.

    Returns:
    float: Total training time in seconds.
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Helper function to make predictions and plot images
def pred_and_plot_image(
    model: nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Loads an image, makes a prediction using the model, and plots it with prediction label.

    Args:
    model (nn.Module): Trained model.
    image_path (str): Path to the image to predict on.
    class_names (List[str], optional): List of class names for the dataset.
    transform (optional): Transformation to apply to the image before prediction.
    device (torch.device, optional): Device to run inference on.
    """
    target_image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0
    if transform:
        target_image = transform(target_image)
    model.to(device)
    model.eval()
    with torch.no_grad():
        target_image = target_image.unsqueeze(0).to(device)
        output = model(target_image)
        pred_probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)

    plt.imshow(target_image.squeeze().permute(1, 2, 0))
    if class_names:
        plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_probs.max():.3f}")
    else:
        plt.title(f"Pred: {pred_label.item()} | Prob: {pred_probs.max():.3f}")
    plt.axis(False)


# Helper function to set random seeds
def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
    seed (int): Seed value for random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Download and unzip data function
def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """
    Downloads and unzips a dataset.

    Args:
    source (str): URL to download the zip file from.
    destination (str): Directory to save the extracted files.
    remove_source (bool): Whether to delete the zip file after extraction.

    Returns:
    Path: Path to the extracted dataset.
    """
    data_path = Path("data/")
    image_path = data_path / destination
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Creating {image_path} directory...")
        image_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file}...")
            f.write(request.content)
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            zip_ref.extractall(image_path)
        if remove_source:
            os.remove(data_path / target_file)
    return image_path


# Customizing color cycle with monochrome settings for clarity in black-and-white printing
mark_every = 0.1
monochrome = (
    cycler("color", ["k"])
    * cycler("markevery", [mark_every])
    * cycler("marker", ["", "o", "^", "s", "v"])
    * cycler("linestyle", ["-", "--", ":", (0, (5, 2, 5, 5, 1, 4))])
)

plt.rc("axes", prop_cycle=monochrome)

# Matplotlib configurations for better visualization aesthetics
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
        "figure.figsize": [12, 8],
        "text.usetex": False,
        "figure.facecolor": "white",
        "figure.autolayout": True,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.formatter.limits": (0, 5),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.loc": "best",
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "xtick.labelsize": 14,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.labelsize": 14,
        "ytick.direction": "in",
        "ytick.right": True,
        "grid.color": "grey",
        "grid.linestyle": "--",
        "grid.linewidth": 0.75,
        "errorbar.capsize": 4,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.4,
        "image.cmap": "viridis",
        "lines.antialiased": True,
        "patch.antialiased": True,
        "text.antialiased": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelpad": 10,
        "axes.titlepad": 15,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.9,
    }
)
