import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torchvision.models as models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt

# Set matplotlib configurations
plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 14,
        "figure.figsize": [15, 8],
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.75",
        "legend.fontsize": "medium",
        "legend.fancybox": False,
        "legend.frameon": False,
        "legend.shadow": False,
        "savefig.transparent": True,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.dpi": 400,
    }
)

# Set data paths
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Extracting disease folders from the training directory
diseases = os.listdir(train_dir)

plant_names = []
disease_names = []
healthy_labels = []

# Separating plant names and disease names, including a separate category for 'healthy'
for disease_folder in diseases:
    parts = disease_folder.split("___")
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
for disease_folder in diseases:
    disease_path = os.path.join(train_dir, disease_folder)
    disease_count[disease_folder] = len(os.listdir(disease_path))

# Convert the disease_count dictionary to a pandas DataFrame for better analysis and visualization
disease_count_df = pd.DataFrame(
    disease_count.values(), index=disease_count.keys(), columns=["no_of_images"]
)

print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases (excluding healthy): {len(disease_names)}")
print(f"Total classes (including healthy labels per plant): {len(diseases)}")
print(f"Total number of images: {sum(disease_count.values())}")
print(disease_count_df)

# Create a dataframe for counting how many images are available for each crop and disease in the training and validation
train_data = []
valid_data = []

for disease in diseases:
    train_data.append(
        {
            "plant": disease.split("___")[0],
            "disease": disease.split("___")[1],
            "no_of_images": len(os.listdir(os.path.join(train_dir, disease))),
        }
    )

    valid_data.append(
        {
            "plant": disease.split("___")[0],
            "disease": disease.split("___")[1],
            "no_of_images": len(os.listdir(os.path.join(valid_dir, disease))),
        }
    )

train_data = pd.DataFrame(train_data)
valid_data = pd.DataFrame(valid_data)

train_data = train_data.groupby(["plant", "disease"]).sum().reset_index()
valid_data = valid_data.groupby(["plant", "disease"]).sum().reset_index()

train_data["data"] = "train"
valid_data["data"] = "valid"

data = pd.concat([train_data, valid_data])

data = data.pivot(
    index=["plant", "disease"], columns="data", values="no_of_images"
).reset_index()

data = data.fillna(0)

data["total_images"] = data.train + data.valid

data = data.sort_values(by="total_images", ascending=False)

data = data.reset_index(drop=True)

data.to_csv("../../data/processed/data.csv", index=False)


class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
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

            if plant not in self.class_to_idx["crop_type"]:
                self.class_to_idx["crop_type"][plant] = len(
                    self.class_to_idx["crop_type"]
                )
                self.idx_to_class["crop_type"][
                    len(self.idx_to_class["crop_type"])
                ] = plant

            if disease not in self.class_to_idx["disease"]:
                self.class_to_idx["disease"][disease] = len(
                    self.class_to_idx["disease"]
                )
                self.idx_to_class["disease"][
                    len(self.idx_to_class["disease"])
                ] = disease

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
            "healthy": torch.tensor(self.labels["healthy"][idx], dtype=torch.float),
        }

        return img, labels


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

print(train_dataset.idx_to_class)

# Define the vision transformer model for multi-task learning from scratch using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformerForMultiTask(nn.Module):
    def __


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
        crop_target, disease_detection_target, disease_classification_target = (
            targets["crop_type"],
            targets["healthy"],
            targets["disease"],
        )

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


# Initialize the model and loss function
model = VisionTransformerForMultiTask()
criterion = MultiTaskLoss()

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10):
    model.to(device)
    model.train()

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), {
                k: v.to(device) for k, v in labels.items()
            }

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        for i, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), {
                k: v.to(device) for k, v in labels.items()
            }

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Time: {time.time()-start_time:.2f}s"
        )

    return train_losses, valid_losses


train_losses, valid_losses = train(
    model, criterion, optimizer, train_loader, valid_loader, num_epochs=10
)

# Plot the training and validation losses
