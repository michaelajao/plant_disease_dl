# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds

# ================================================================
# Configuration and Settings
# ================================================================
# Set seeds for reproducibility
set_seeds(42)

# Hyperparameters and configuration setup
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
HEIGHT, WIDTH = 224, 224

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

# Extract disease folders from the training directory
disease_folders = os.listdir(train_dir)

# Initialize lists to store plant and disease names
plant_names = []
disease_names = []
healthy_labels = []

# Separating plant names and disease names, including a separate category for 'Healthy'
for folder_name in disease_folders:
    parts = folder_name.split("___")
    plant = parts[0]
    disease = parts[1] if len(parts) > 1 else "Healthy"

    if plant not in plant_names:
        plant_names.append(plant)

    if disease == "Healthy":
        healthy_labels.append(f"{plant}___Healthy")
    elif disease not in disease_names:
        disease_names.append(disease)

# Count the number of images for each disease in the dataset
disease_count = {}
for folder_name in disease_folders:
    folder_path = os.path.join(train_dir, folder_name)
    disease_count[folder_name] = len(os.listdir(folder_path))

# Convert the disease_count dictionary to a pandas DataFrame
disease_count_df = pd.DataFrame(
    disease_count.items(), columns=["Disease", "No_of_images"]
)
disease_count_df = disease_count_df.sort_values(by="No_of_images", ascending=False)

print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases (excluding healthy): {len(disease_names)}")
print(f"Total classes (including healthy labels per plant): {len(disease_folders)}")

# Display the DataFrame for analysis
print(disease_count_df)

# Save the disease count DataFrame
disease_count_df.to_csv("../../reports/results/disease_counts.csv", index=False)

# ================================================================
# Creating DataFrames for Training and Validation Data
# ================================================================
train_data_list = []
valid_data_list = []

for folder_name in disease_folders:
    plant = folder_name.split("___")[0]
    disease = folder_name.split("___")[1] if "___" in folder_name else "Healthy"

    train_num_images = len(os.listdir(os.path.join(train_dir, folder_name)))
    valid_num_images = len(os.listdir(os.path.join(valid_dir, folder_name)))

    train_data_list.append(
        {"plant": plant, "disease": disease, "no_of_images": train_num_images}
    )
    valid_data_list.append(
        {"plant": plant, "disease": disease, "no_of_images": valid_num_images}
    )

train_data_df = pd.DataFrame(train_data_list)
valid_data_df = pd.DataFrame(valid_data_list)

train_data_df["data"] = "train"
valid_data_df["data"] = "valid"

data_df = pd.concat([train_data_df, valid_data_df])

data_pivot = data_df.pivot_table(
    index=["plant", "disease"], columns="data", values="no_of_images", fill_value=0
).reset_index()

data_pivot["total_images"] = data_pivot["train"] + data_pivot["valid"]
data_pivot = data_pivot.sort_values(by="total_images", ascending=False).reset_index(
    drop=True
)

# Save the combined data DataFrame
if not os.path.exists("../../data/processed"):
    os.makedirs("../../data/processed")
data_pivot.to_csv("../../data/processed/data_summary.csv", index=False)

# ================================================================
# Custom Dataset Definition
# ================================================================
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for plant disease classification that handles
        - Crop Type Classification
        - Disease Detection (Healthy vs Diseased)
        - Disease Type Classification
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = {"crop_type": [], "disease": [], "healthy": []}
        self.class_to_idx = {
            "crop_type": {},
            "disease": {},
            "healthy": {True: 1, False: 0},
        }
        self.idx_to_class = {
            "crop_type": {},
            "disease": {},
            "healthy": {1: True, 0: False},
        }

        self._prepare_dataset()

    def _prepare_dataset(self):
        disease_folders = os.listdir(self.data_dir)
        for folder_name in disease_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            images = os.listdir(folder_path)
            plant, disease = folder_name.split("___")

            # Map crop types
            if plant not in self.class_to_idx["crop_type"]:
                idx = len(self.class_to_idx["crop_type"])
                self.class_to_idx["crop_type"][plant] = idx
                self.idx_to_class["crop_type"][idx] = plant

            # Map diseases
            if disease not in self.class_to_idx["disease"]:
                idx = len(self.class_to_idx["disease"])
                self.class_to_idx["disease"][disease] = idx
                self.idx_to_class["disease"][idx] = disease

            for img in images:
                img_path = os.path.join(folder_path, img)
                self.data.append(img_path)
                self.labels["crop_type"].append(self.class_to_idx["crop_type"][plant])
                self.labels["disease"].append(self.class_to_idx["disease"][disease])
                self.labels["healthy"].append(1 if disease == "Healthy" else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = datasets.folder.default_loader(img_path)
        if self.transform:
            img = self.transform(img)

        labels = {
            "crop_type": torch.tensor(self.labels["crop_type"][idx]),
            "disease": torch.tensor(self.labels["disease"][idx]),
            "healthy": torch.tensor(self.labels["healthy"][idx], dtype=torch.float32),
        }

        return img, labels

# ================================================================
# Data Transforms and Data Loaders
# ================================================================
# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
)

# Access class mappings
print("Crop Type Classes:", train_dataset.class_to_idx["crop_type"])
print("Disease Classes:", train_dataset.class_to_idx["disease"])

# ================================================================
# Model Definitions
# ================================================================
# ------------------------------
# MultiTaskCNN Model Definition
# ------------------------------
class MultiTaskCNN(nn.Module):
    def __init__(self, num_crop_types, num_diseases):
        super(MultiTaskCNN, self).__init__()

        # Load a pre-trained model as feature extractor
        self.feature_extractor = models.resnet50(pretrained=True)

        # Remove the last layer (fully connected layer) of the feature extractor
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        # Task-specific layers
        # Crop classification head
        self.crop_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_crop_types),
        )

        # Disease detection head (binary classification: healthy vs diseased)
        self.disease_detection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Disease classification head
        self.disease_classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        # Shared feature extraction
        x = self.feature_extractor(x)

        # Task-specific predictions
        crop_pred = self.crop_head(x)
        disease_detection_pred = self.disease_detection_head(x)
        disease_classification_pred = self.disease_classification_head(x)

        return crop_pred, disease_detection_pred, disease_classification_pred

# ------------------------------
# MultiTaskViT Model Definition (Implemented from Scratch)
# ------------------------------
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # shape: (batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # shape: (batch_size, num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = x + self.pos_embed
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection

        # Feed-forward network
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # Residual connection
        return x

class MultiTaskViT(nn.Module):
    def __init__(
        self,
        num_crop_types,
        num_diseases,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super(MultiTaskViT, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_embed = PositionalEncoding(seq_len=num_patches + 1, embed_dim=embed_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Task-specific heads
        # Crop classification head
        self.crop_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_crop_types),
        )

        # Disease detection head (binary classification: healthy vs diseased)
        self.disease_detection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Disease classification head
        self.disease_classification_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Concatenate class token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = self.pos_embed(x)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x)

        # Take the class token output
        x = self.norm(x)
        cls_output = x[:, 0]

        # Task-specific predictions
        crop_pred = self.crop_head(cls_output)
        disease_detection_pred = self.disease_detection_head(cls_output)
        disease_classification_pred = self.disease_classification_head(cls_output)

        return crop_pred, disease_detection_pred, disease_classification_pred

# ================================================================
# Loss Function Definition
# ================================================================
class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_crop=1.0,
        weight_disease_detection=1.0,
        weight_disease_classification=1.0,
    ):
        super(MultiTaskLoss, self).__init__()
        self.weight_crop = weight_crop
        self.weight_disease_detection = weight_disease_detection
        self.weight_disease_classification = weight_disease_classification
        self.loss_crop = nn.CrossEntropyLoss()
        self.loss_disease_detection = nn.BCELoss()
        self.loss_disease_classification = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Unpack the outputs and targets
        crop_pred, disease_detection_pred, disease_classification_pred = outputs
        crop_target = targets["crop_type"]
        disease_detection_target = targets["healthy"]
        disease_classification_target = targets["disease"]

        # Calculate individual losses
        loss_crop = self.loss_crop(crop_pred, crop_target)
        loss_disease_detection = self.loss_disease_detection(
            disease_detection_pred.view(-1), disease_detection_target
        )
        loss_disease_classification = self.loss_disease_classification(
            disease_classification_pred, disease_classification_target
        )

        # Weighted sum of the individual losses
        total_loss = (
            self.weight_crop * loss_crop
            + self.weight_disease_detection * loss_disease_detection
            + self.weight_disease_classification * loss_disease_classification
        )

        return total_loss

# ================================================================
# Model Instantiation
# ================================================================
num_crop_types = len(train_dataset.class_to_idx["crop_type"])
num_diseases = len(train_dataset.class_to_idx["disease"])

# Instantiate MultiTaskCNN
cnn_model = MultiTaskCNN(num_crop_types, num_diseases)

# Instantiate MultiTaskViT from scratch
vit_model = MultiTaskViT(
    num_crop_types=num_crop_types,
    num_diseases=num_diseases,
    img_size=HEIGHT,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
)

# Move models to device and wrap with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    cnn_model = nn.DataParallel(cnn_model)
    vit_model = nn.DataParallel(vit_model)

cnn_model.to(device)
vit_model.to(device)

# ================================================================
# Optimizer, Loss Function, and Scheduler
# ================================================================
criterion = MultiTaskLoss().to(device)

# Optimizer and Scheduler for CNN model
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    cnn_optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

# Optimizer and Scheduler for ViT model
vit_optimizer = optim.Adam(vit_model.parameters(), lr=LEARNING_RATE)
vit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    vit_optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

# ================================================================
# Early Stopping Class Definition
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
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# ================================================================
# Training and Evaluation Functions
# ================================================================
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs=NUM_EPOCHS,
    model_name="Model",
    max_grad_norm=None,
):
    """
    Training loop for the multi-task model.
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.0)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_crop = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs} - {model_name}"):
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            running_train_loss += loss.item()

            crop_pred, _, _ = outputs
            _, crop_pred_labels = torch.max(crop_pred, 1)
            correct_crop += (crop_pred_labels == labels["crop_type"]).sum().item()
            total += labels["crop_type"].size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_crop / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_crop_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc=f"Validation - {model_name}"):
                inputs = inputs.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                crop_pred, _, _ = outputs
                _, crop_pred_labels = torch.max(crop_pred, 1)
                correct_crop_val += (crop_pred_labels == labels["crop_type"]).sum().item()
                total_val += labels["crop_type"].size(0)

        avg_val_loss = running_val_loss / len(valid_loader)
        val_accuracy = correct_crop_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"{model_name} Train Loss: {avg_train_loss:.4f} "
            f"Train Acc: {train_accuracy*100:.2f}% "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Acc: {val_accuracy*100:.2f}%"
        )

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return model, train_losses, val_losses, train_accuracies, val_accuracies

# ================================================================
# Training the Models
# ================================================================
print("Training MultiTaskCNN Model")
cnn_trained_model, cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies = train_model(
    cnn_model,
    criterion,
    cnn_optimizer,
    cnn_scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs=NUM_EPOCHS,
    model_name="MultiTaskCNN",
    max_grad_norm=1.0,  # You can adjust this value
)

print("\nTraining MultiTaskViT Model")
vit_trained_model, vit_train_losses, vit_val_losses, vit_train_accuracies, vit_val_accuracies = train_model(
    vit_model,
    criterion,
    vit_optimizer,
    vit_scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs=NUM_EPOCHS,
    model_name="MultiTaskViT",
    max_grad_norm=1.0,  # Gradient clipping helps when training from scratch
)

# ================================================================
# Plotting Training and Validation Metrics
# ================================================================
# Function to plot metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_loss.pdf")
    plt.close()

    # Plot the training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(
        [acc * 100 for acc in train_accuracies], label="Training Accuracy", color="blue"
    )
    plt.plot(
        [acc * 100 for acc in val_accuracies], label="Validation Accuracy", color="orange"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training and Validation Accuracy - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_accuracy.pdf")
    plt.close()

# Plot metrics for CNN model
plot_metrics(cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies, "MultiTaskCNN")

# Plot metrics for ViT model
plot_metrics(vit_train_losses, vit_val_losses, vit_train_accuracies, vit_val_accuracies, "MultiTaskViT")

# ================================================================
# Evaluation Function
# ================================================================
def evaluate(model, data_loader, device):
    model.eval()
    correct_crop = 0
    total = 0
    all_crop_preds = []
    all_crop_targets = []

    correct_disease = 0
    all_disease_preds = []
    all_disease_targets = []

    correct_healthy = 0
    all_healthy_preds = []
    all_healthy_targets = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            outputs = model(inputs)

            # Crop Type Evaluation
            crop_pred, disease_detection_pred, disease_classification_pred = outputs
            _, crop_pred_labels = torch.max(crop_pred, 1)
            correct_crop += (crop_pred_labels == labels["crop_type"]).sum().item()
            total += labels["crop_type"].size(0)
            all_crop_preds.extend(crop_pred_labels.cpu().numpy())
            all_crop_targets.extend(labels["crop_type"].cpu().numpy())

            # Disease Detection Evaluation
            healthy_pred_labels = (disease_detection_pred > 0.5).float().view(-1)
            correct_healthy += (
                healthy_pred_labels == labels["healthy"]
            ).sum().item()
            all_healthy_preds.extend(healthy_pred_labels.cpu().numpy())
            all_healthy_targets.extend(labels["healthy"].cpu().numpy())

            # Disease Classification Evaluation
            _, disease_pred_labels = torch.max(disease_classification_pred, 1)
            correct_disease += (
                disease_pred_labels == labels["disease"]
            ).sum().item()
            all_disease_preds.extend(disease_pred_labels.cpu().numpy())
            all_disease_targets.extend(labels["disease"].cpu().numpy())

    crop_accuracy = correct_crop / total
    healthy_accuracy = correct_healthy / total
    disease_accuracy = correct_disease / total

    results = {
        "crop": {
            "accuracy": crop_accuracy,
            "predictions": all_crop_preds,
            "targets": all_crop_targets,
        },
        "healthy": {
            "accuracy": healthy_accuracy,
            "predictions": all_healthy_preds,
            "targets": all_healthy_targets,
        },
        "disease": {
            "accuracy": disease_accuracy,
            "predictions": all_disease_preds,
            "targets": all_disease_targets,
        },
    }

    return results

# ================================================================
# Evaluating the Models
# ================================================================
print("\nEvaluating MultiTaskCNN Model")
cnn_train_results = evaluate(cnn_trained_model, train_loader, device)
cnn_valid_results = evaluate(cnn_trained_model, valid_loader, device)

print("\nEvaluating MultiTaskViT Model")
vit_train_results = evaluate(vit_trained_model, train_loader, device)
vit_valid_results = evaluate(vit_trained_model, valid_loader, device)

# ================================================================
# Displaying Evaluation Metrics
# ================================================================
def display_metrics(train_results, valid_results, model_name):
    print(f"\n{model_name} Training Crop Type Accuracy: {train_results['crop']['accuracy']*100:.2f}%")
    print(f"{model_name} Validation Crop Type Accuracy: {valid_results['crop']['accuracy']*100:.2f}%")

    # Additional Metrics for Crop Type Classification
    train_precision = precision_score(
        train_results["crop"]["targets"],
        train_results["crop"]["predictions"],
        average="macro",
    )
    train_recall = recall_score(
        train_results["crop"]["targets"],
        train_results["crop"]["predictions"],
        average="macro",
    )
    train_f1 = f1_score(
        train_results["crop"]["targets"],
        train_results["crop"]["predictions"],
        average="macro",
    )

    valid_precision = precision_score(
        valid_results["crop"]["targets"],
        valid_results["crop"]["predictions"],
        average="macro",
    )
    valid_recall = recall_score(
        valid_results["crop"]["targets"],
        valid_results["crop"]["predictions"],
        average="macro",
    )
    valid_f1 = f1_score(
        valid_results["crop"]["targets"],
        valid_results["crop"]["predictions"],
        average="macro",
    )

    print(
        f"{model_name} Training Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}"
    )
    print(
        f"{model_name} Validation Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1 Score: {valid_f1:.4f}"
    )

# Display metrics for CNN model
display_metrics(cnn_train_results, cnn_valid_results, "MultiTaskCNN")

# Display metrics for ViT model
display_metrics(vit_train_results, vit_valid_results, "MultiTaskViT")

# ================================================================
# Confusion Matrix for Crop Type Classification
# ================================================================
def plot_confusion_matrix(targets, predictions, classes, model_name, task_name):
    cm = confusion_matrix(targets, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {task_name} - {model_name}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{model_name}_{task_name}_confusion_matrix.pdf")
    plt.close()

crop_classes = list(train_dataset.class_to_idx["crop_type"].keys())         

# Confusion matrix for CNN model
plot_confusion_matrix(
    cnn_valid_results["crop"]["targets"],
    cnn_valid_results["crop"]["predictions"],
    crop_classes,
    "MultiTaskCNN",
    "Crop_Type",
)

# Confusion matrix for ViT model
plot_confusion_matrix(
    vit_valid_results["crop"]["targets"],
    vit_valid_results["crop"]["predictions"],           
    crop_classes,
    "MultiTaskViT",
    "Crop_Type",
)

# ================================================================
# Saving the Trained Models
# ================================================================
torch.save(cnn_trained_model.state_dict(), "../../models/MultiTaskCNN.pth")
torch.save(vit_trained_model.state_dict(), "../../models/MultiTaskViT.pth")
