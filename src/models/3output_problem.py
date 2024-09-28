# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds  # Custom helper functions

# ================================================================
# Configuration and Settings
# ================================================================
# Set seeds for reproducibility
set_seeds(42)

# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# ================================================================
# Data Preprocessing and Exploration
# ================================================================

# Function to extract plant and disease names and count images per class
def get_data_info(data_dir):
    disease_folders = os.listdir(data_dir)
    plant_names = set()
    disease_names = set()
    healthy_labels = []
    disease_counts = {}

    for folder_name in disease_folders:
        folder_path = os.path.join(data_dir, folder_name)
        num_images = len(os.listdir(folder_path))
        parts = folder_name.split("___")
        plant = parts[0]
        disease = parts[1] if len(parts) > 1 else "Healthy"

        plant_names.add(plant)
        if disease == "Healthy":
            healthy_labels.append(f"{plant}___Healthy")
        else:
            disease_names.add(disease)

        disease_counts[folder_name] = num_images

    return {
        "plant_names": sorted(list(plant_names)),
        "disease_names": sorted(list(disease_names)),
        "healthy_labels": healthy_labels,
        "disease_counts": disease_counts,
    }

# Get data information from training and validation directories
train_info = get_data_info(train_dir)
valid_info = get_data_info(valid_dir)

# Print dataset statistics
print(f"Number of unique plants: {len(train_info['plant_names'])}")
print(f"Number of unique diseases (excluding healthy): {len(train_info['disease_names'])}")
print(f"Total classes (including healthy labels per plant): {len(train_info['disease_counts'])}")
print(f"Total number of images in training data: {sum(train_info['disease_counts'].values())}")

# Create DataFrames for training and validation data counts
def create_data_count_df(data_info, data_type):
    data_list = []
    for folder_name, num_images in data_info['disease_counts'].items():
        plant, disease = folder_name.split("___") if "___" in folder_name else (folder_name, "Healthy")
        data_list.append({
            'Plant': plant,
            'Disease': disease,
            'Number of Images': num_images,
            'Data': data_type
        })
    return pd.DataFrame(data_list)

# Combine training and validation data counts
train_data_df = create_data_count_df(train_info, 'Train')
valid_data_df = create_data_count_df(valid_info, 'Valid')
data_counts_df = pd.concat([train_data_df, valid_data_df], ignore_index=True)

# Pivot the DataFrame for better visualization
data_pivot = data_counts_df.pivot_table(
    index=['Plant', 'Disease'],
    columns='Data',
    values='Number of Images',
    fill_value=0
).reset_index()

# Calculate total images per class
data_pivot['Total Images'] = data_pivot['Train'] + data_pivot['Valid']

# Save the data counts to a CSV file
data_pivot.to_csv("../../data/processed/data_counts.csv", index=False)

# ================================================================
# Custom Dataset Class: PlantDiseaseDataset
# ================================================================

class PlantDiseaseDataset(Dataset):
    """
    Custom dataset for plant disease classification that handles:
    - Crop Type Classification
    - Disease Detection (Healthy vs Diseased)
    - Disease Type Classification
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): The directory path where the dataset is located.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = {
            'crop_type': [],
            'disease': [],
            'healthy': []
        }
        self.class_to_idx = {
            'crop_type': {},
            'disease': {},
            'healthy': {True: 1, False: 0}
        }
        self.idx_to_class = {
            'crop_type': {},
            'disease': {},
            'healthy': {1: True, 0: False}
        }
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Prepares the dataset by populating image paths and labels.
        """
        disease_folders = os.listdir(self.data_dir)
        for folder_name in disease_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            images = os.listdir(folder_path)
            plant, disease = folder_name.split("___") if "___" in folder_name else (folder_name, "Healthy")

            # Update class_to_idx and idx_to_class dictionaries for crop_type
            if plant not in self.class_to_idx['crop_type']:
                idx = len(self.class_to_idx['crop_type'])
                self.class_to_idx['crop_type'][plant] = idx
                self.idx_to_class['crop_type'][idx] = plant

            # Update class_to_idx and idx_to_class dictionaries for disease
            if disease not in self.class_to_idx['disease']:
                idx = len(self.class_to_idx['disease'])
                self.class_to_idx['disease'][disease] = idx
                self.idx_to_class['disease'][idx] = disease

            # Process images and labels
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels['crop_type'].append(self.class_to_idx['crop_type'][plant])
                self.labels['disease'].append(self.class_to_idx['disease'][disease])
                self.labels['healthy'].append(1 if disease == "Healthy" else 0)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and labels at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, labels) where labels is a dictionary.
        """
        img_path = self.image_paths[idx]
        image = datasets.folder.default_loader(img_path)  # PIL Image
        if self.transform:
            image = self.transform(image)

        labels = {
            'crop_type': torch.tensor(self.labels['crop_type'][idx], dtype=torch.long),
            'disease': torch.tensor(self.labels['disease'][idx], dtype=torch.long),
            'healthy': torch.tensor(self.labels['healthy'][idx], dtype=torch.float32)
        }
        return image, labels

# ================================================================
# Data Transforms and Data Loaders
# ================================================================

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Uncomment the normalization if required
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

# Create dataset instances
train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Print class mappings
print("Class to Index Mapping:")
print(train_dataset.class_to_idx)
print("\nIndex to Class Mapping:")
print(train_dataset.idx_to_class)

# ================================================================
# Visualization: Sample Images with Labels
# ================================================================

def plot_multitask_samples(dataset, filename):
    """
    Plots sample images with their labels from the dataset.
    Args:
        dataset (Dataset): The dataset to sample images from.
        filename (str): The filename to save the plot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(10):
        idx = np.random.randint(len(dataset))
        img, labels = dataset[idx]
        plant = dataset.idx_to_class['crop_type'][labels['crop_type'].item()]
        disease = dataset.idx_to_class['disease'][labels['disease'].item()]
        healthy = 'Healthy' if labels['healthy'].item() == 1 else 'Diseased'

        ax = axes[i // 5, i % 5]
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Plant: {plant}\nDisease: {disease}\nHealthy: {healthy}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot and save sample images
plot_multitask_samples(train_dataset, "../../reports/figures/sample_images_multitask_problem.pdf")

# ================================================================
# Improved Multi-Task CNN Model Definition
# ================================================================

class MultiTaskCNN(nn.Module):
    """
    Improved Multi-task CNN model that predicts:
    - Crop Type Classification
    - Disease Detection (Healthy vs Diseased)
    - Disease Type Classification
    """
    def __init__(self, num_crop_types, num_diseases):
        """
        Args:
            num_crop_types (int): Number of crop types.
            num_diseases (int): Number of disease types.
        """
        super(MultiTaskCNN, self).__init__()

        # Load a pre-trained EfficientNet-B0 model as the feature extractor
        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        self.feature_extractor.classifier = nn.Identity()  # Remove the classifier

        num_features = self.feature_extractor.classifier.in_features

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Crop classification head
        self.crop_head = nn.Sequential(
            nn.Linear(512, num_crop_types)
        )

        # Disease detection head (binary classification: Healthy vs Diseased)
        self.disease_detection_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Disease classification head
        self.disease_classification_head = nn.Sequential(
            nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            tuple: Outputs for each task.
        """
        # Shared feature extraction
        features = self.feature_extractor(x)
        shared_output = self.shared_fc(features)

        # Task-specific predictions
        crop_pred = self.crop_head(shared_output)
        disease_detection_pred = self.disease_detection_head(shared_output)
        disease_classification_pred = self.disease_classification_head(shared_output)

        return crop_pred, disease_detection_pred, disease_classification_pred

# ================================================================
# Improved Multi-Task Loss Function
# ================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines losses for:
    - Crop Type Classification
    - Disease Detection
    - Disease Type Classification
    """
    def __init__(self, task_weights=None):
        """
        Args:
            task_weights (dict): Weights for each task's loss.
        """
        super(MultiTaskLoss, self).__init__()
        if task_weights is None:
            task_weights = {'crop': 1.0, 'disease_detection': 1.0, 'disease_classification': 1.0}
        self.task_weights = task_weights

        self.loss_crop = nn.CrossEntropyLoss()
        self.loss_disease_detection = nn.BCELoss()
        self.loss_disease_classification = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Calculates the combined loss.
        Args:
            outputs (tuple): Outputs from the model.
            targets (dict): Dictionary of target tensors.
        Returns:
            Tensor: Combined loss.
        """
        # Unpack outputs and targets
        crop_pred, disease_detection_pred, disease_classification_pred = outputs
        crop_target = targets['crop_type']
        disease_detection_target = targets['healthy']
        disease_classification_target = targets['disease']

        # Calculate individual losses
        loss_crop = self.loss_crop(crop_pred, crop_target)
        loss_disease_detection = self.loss_disease_detection(
            disease_detection_pred.squeeze(), disease_detection_target
        )
        loss_disease_classification = self.loss_disease_classification(
            disease_classification_pred, disease_classification_target
        )

        # Weighted sum of individual losses
        total_loss = (
            self.task_weights['crop'] * loss_crop +
            self.task_weights['disease_detection'] * loss_disease_detection +
            self.task_weights['disease_classification'] * loss_disease_classification
        )

        return total_loss, {
            'loss_crop': loss_crop.item(),
            'loss_disease_detection': loss_disease_detection.item(),
            'loss_disease_classification': loss_disease_classification.item()
        }

# ================================================================
# Evaluation Function
# ================================================================

def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation data.
    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run the evaluation on.
    Returns:
        dict: Dictionary containing accuracies and losses for each task.
    """
    model.eval()
    total_correct = {'crop': 0, 'disease_detection': 0, 'disease_classification': 0}
    total_samples = 0
    total_loss = 0.0
    all_targets = {'crop': [], 'disease_detection': [], 'disease_classification': []}
    all_predictions = {'crop': [], 'disease_detection': [], 'disease_classification': []}

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            batch_size = images.size(0)

            outputs = model(images)
            crop_pred, disease_detection_pred, disease_classification_pred = outputs

            # Compute losses (without backpropagation)
            criterion = MultiTaskLoss()
            loss, _ = criterion(outputs, labels)
            total_loss += loss.item() * batch_size

            # Predictions
            crop_pred_labels = crop_pred.argmax(dim=1)
            disease_detection_pred_labels = (disease_detection_pred.squeeze() > 0.5).long()
            disease_classification_pred_labels = disease_classification_pred.argmax(dim=1)

            # Update total correct predictions
            total_correct['crop'] += (crop_pred_labels == labels['crop_type']).sum().item()
            total_correct['disease_detection'] += (disease_detection_pred_labels == labels['healthy']).sum().item()
            total_correct['disease_classification'] += (disease_classification_pred_labels == labels['disease']).sum().item()

            # Append targets and predictions for metrics
            total_samples += batch_size
            all_targets['crop'].extend(labels['crop_type'].cpu().numpy())
            all_targets['disease_detection'].extend(labels['healthy'].cpu().numpy())
            all_targets['disease_classification'].extend(labels['disease'].cpu().numpy())

            all_predictions['crop'].extend(crop_pred_labels.cpu().numpy())
            all_predictions['disease_detection'].extend(disease_detection_pred_labels.cpu().numpy())
            all_predictions['disease_classification'].extend(disease_classification_pred_labels.cpu().numpy())

    # Calculate accuracies
    accuracies = {
        'crop': total_correct['crop'] / total_samples,
        'disease_detection': total_correct['disease_detection'] / total_samples,
        'disease_classification': total_correct['disease_classification'] / total_samples
    }
    avg_loss = total_loss / total_samples

    return accuracies, avg_loss, all_predictions, all_targets

# ================================================================
# Early Stopping Class
# ================================================================

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training is stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Checks if validation loss has improved; if not, increments counter.
        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0  # Reset counter if validation loss improves

# ================================================================
# Training Loop Function
# ================================================================

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=10
):
    """
    Trains the model and evaluates on validation data.
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        num_epochs (int): Number of epochs.
    Returns:
        dict: Dictionary containing training history.
    """
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_accuracies': [],
        'valid_accuracies': [],
    }
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n{'-' * 30}")
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct_preds = {'crop': 0, 'disease_detection': 0, 'disease_classification': 0}

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            batch_size = images.size(0)
            total_samples += batch_size

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size

            # Predictions
            crop_pred, disease_detection_pred, disease_classification_pred = outputs
            crop_pred_labels = crop_pred.argmax(dim=1)
            disease_detection_pred_labels = (disease_detection_pred.squeeze() > 0.5).long()
            disease_classification_pred_labels = disease_classification_pred.argmax(dim=1)

            # Update correct predictions
            correct_preds['crop'] += (crop_pred_labels == labels['crop_type']).sum().item()
            correct_preds['disease_detection'] += (disease_detection_pred_labels == labels['healthy']).sum().item()
            correct_preds['disease_classification'] += (disease_classification_pred_labels == labels['disease']).sum().item()

        epoch_loss = running_loss / total_samples
        train_accuracies = {
            'crop': correct_preds['crop'] / total_samples,
            'disease_detection': correct_preds['disease_detection'] / total_samples,
            'disease_classification': correct_preds['disease_classification'] / total_samples
        }
        history['train_loss'].append(epoch_loss)
        history['train_accuracies'].append(train_accuracies)

        # Validation phase
        model.eval()
        valid_accuracies, valid_loss, _, _ = evaluate(model, valid_loader, device)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracies'].append(valid_accuracies)

        scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Train Loss: {epoch_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Train Acc Crop: {train_accuracies['crop'] * 100:.2f}% | "
            f"Valid Acc Crop: {valid_accuracies['crop'] * 100:.2f}% | "
            f"Time: {int(epoch_mins)}m {int(epoch_secs)}s | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early Stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return history

# ================================================================
# Model Training Configuration
# ================================================================

# Instantiate the model
num_crop_types = len(train_dataset.class_to_idx['crop_type'])
num_diseases = len(train_dataset.class_to_idx['disease'])
model = MultiTaskCNN(num_crop_types, num_diseases).to(device)

# Define loss function and optimizer
task_weights = {'crop': 1.0, 'disease_detection': 1.0, 'disease_classification': 1.0}
criterion = MultiTaskLoss(task_weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ================================================================
# Train the Model
# ================================================================

# Train the model
history = train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=10  # Adjust the number of epochs as needed
)

# ================================================================
# Plot Training and Validation Losses and Accuracies
# ================================================================

def plot_training_history(history, filename):
    """
    Plots training and validation losses and accuracies.
    Args:
        history (dict): Training history returned by train_model.
        filename (str): Filename to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot losses
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Plot accuracies for crop classification
    plt.subplot(1, 2, 2)
    train_acc_crop = [acc['crop'] for acc in history['train_accuracies']]
    valid_acc_crop = [acc['crop'] for acc in history['valid_accuracies']]
    plt.plot(epochs, train_acc_crop, label='Train Accuracy (Crop)')
    plt.plot(epochs, valid_acc_crop, label='Valid Accuracy (Crop)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch (Crop Classification)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot and save training history
plot_training_history(history, "../../reports/figures/training_results_multitask_problem.pdf")

# ================================================================
# Save and Load the Trained Model
# ================================================================

# Save the trained model
torch.save(model.state_dict(), "../../models/multitask_model.pth")
print("Model saved.")

# Load the trained model (if needed)
# model.load_state_dict(torch.load("../../models/multitask_model.pth"))
# model.to(device)
# print("Model loaded.")

# ================================================================
# Evaluate the Model and Generate Confusion Matrix
# ================================================================

# Evaluate the model on the validation data
accuracies, avg_loss, predictions, targets = evaluate(model, valid_loader, device)
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Validation Accuracies: {accuracies}")

# Generate confusion matrix for disease classification
cm = confusion_matrix(targets['disease_classification'], predictions['disease_classification'])
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[valid_dataset.idx_to_class['disease'][i] for i in range(num_diseases)],
            yticklabels=[valid_dataset.idx_to_class['disease'][i] for i in range(num_diseases)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Disease Classification")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("../../reports/figures/confusion_matrix_multitask_problem.pdf")
plt.show()

# ================================================================
# Calculate Precision, Recall, and F1 Score
# ================================================================

# Disease Classification Metrics
precision = precision_score(
    targets['disease_classification'],
    predictions['disease_classification'],
    average='weighted',
    zero_division=0
)
recall = recall_score(
    targets['disease_classification'],
    predictions['disease_classification'],
    average='weighted',
    zero_division=0
)
f1 = f1_score(
    targets['disease_classification'],
    predictions['disease_classification'],
    average='weighted',
    zero_division=0
)

print(f"Disease Classification Metrics:")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Similarly, compute metrics for Crop Classification and Disease Detection if needed

# ================================================================
# Plot Predictions on Validation Data
# ================================================================

def plot_prediction_results(model, dataset, device, filename):
    """
    Plots sample predictions from the model on the validation dataset.
    Args:
        model (nn.Module): Trained model.
        dataset (Dataset): Validation dataset.
        device (torch.device): Device to run the model on.
        filename (str): Filename to save the plot.
    """
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(30, 15))

    for i in range(10):
        idx = np.random.randint(len(dataset))
        img, labels = dataset[idx]
        img_input = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

        with torch.no_grad():
            crop_pred, disease_detection_pred, disease_classification_pred = model(img_input)

        # Predictions
        crop_pred_label = crop_pred.argmax(dim=1).item()
        disease_detection_prob = disease_detection_pred.item()
        disease_detection_label = 'Healthy' if disease_detection_prob > 0.5 else 'Diseased'
        disease_classification_pred_label = disease_classification_pred.argmax(dim=1).item()
        disease_classification_label = dataset.idx_to_class['disease'][disease_classification_pred_label]

        # Ground truth
        plant = dataset.idx_to_class['crop_type'][labels['crop_type'].item()]
        disease = dataset.idx_to_class['disease'][labels['disease'].item()]
        healthy = 'Healthy' if labels['healthy'].item() == 1 else 'Diseased'

        ax = axes[i // 5, i % 5]
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(
            f"True Plant: {plant}\n"
            f"True Disease: {disease}\n"
            f"True Health Status: {healthy}\n\n"
            f"Predicted Plant: {dataset.idx_to_class['crop_type'][crop_pred_label]}\n"
            f"Predicted Health Status: {disease_detection_label}\n"
            f"Predicted Disease: {disease_classification_label}"
        )

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot and save prediction results
plot_prediction_results(model, valid_dataset, device, "../../reports/figures/prediction_results_multitask_problem.pdf")
