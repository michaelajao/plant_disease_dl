# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from zipfile import ZipFile
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds

# ================================================================
# Configuration and Settings
# ================================================================
# Set seeds for reproducibility
set_seeds(42)

# Hyperparameters and configuration setup
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Directory paths
RAW_DATA_PATH = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
PROCESSED_DATA_PATH = "../../data/processed/"
TRAIN_DIR = os.path.join(RAW_DATA_PATH, "train")
VALID_DIR = os.path.join(RAW_DATA_PATH, "valid")

# Create directories if they don't exist
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# ================================================================
# Data Preparation
# ================================================================
def create_dataframes(train_dir, valid_dir):
    """
    Creates dataframes containing image paths and labels for training and validation datasets.
    """
    # Initialize lists to store data
    data_entries = []

    # Loop through train and valid directories
    for split, directory in [("train", train_dir), ("valid", valid_dir)]:
        # Get all disease folders
        disease_folders = os.listdir(directory)

        for folder_name in disease_folders:
            folder_path = os.path.join(directory, folder_name)
            images = os.listdir(folder_path)

            # Extract plant and disease names
            if "___" in folder_name:
                plant, disease = folder_name.split("___")
            else:
                plant = folder_name
                disease = "Healthy"

            # Loop through all images in the folder
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                data_entries.append(
                    {
                        "split": split,
                        "image_path": img_path,
                        "plant": plant,
                        "disease": disease,
                        "is_healthy": disease == "Healthy",
                    }
                )

    # Create DataFrame
    data_df = pd.DataFrame(data_entries)
    return data_df

# Create DataFrames for train and validation data
data_df = create_dataframes(TRAIN_DIR, VALID_DIR)

# Display the first few rows
print("Data DataFrame:")
print(data_df.head())

# ================================================================
# Label Encoding
# ================================================================
def encode_labels(data_df):
    """
    Encodes categorical labels into numeric labels for model training.
    """
    # Encode 'plant' labels
    plant_types = data_df["plant"].unique()
    plant_to_idx = {plant: idx for idx, plant in enumerate(plant_types)}
    idx_to_plant = {idx: plant for plant, idx in plant_to_idx.items()}
    data_df["plant_label"] = data_df["plant"].map(plant_to_idx)

    # Encode 'disease' labels
    disease_types = data_df["disease"].unique()
    disease_to_idx = {disease: idx for idx, disease in enumerate(disease_types)}
    idx_to_disease = {idx: disease for disease, idx in disease_to_idx.items()}
    data_df["disease_label"] = data_df["disease"].map(disease_to_idx)

    # Encode 'is_healthy' labels (already boolean, convert to int)
    data_df["healthy_label"] = data_df["is_healthy"].astype(int)

    # Save label mappings for future reference
    label_mappings = {
        "plant_to_idx": plant_to_idx,
        "idx_to_plant": idx_to_plant,
        "disease_to_idx": disease_to_idx,
        "idx_to_disease": idx_to_disease,
    }

    return data_df, label_mappings

# Encode labels
data_df, label_mappings = encode_labels(data_df)

# Display label mappings
print("\nLabel Mappings:")
print("Plant to Index:", label_mappings["plant_to_idx"])
print("Disease to Index:", label_mappings["disease_to_idx"])

# Save label mappings to a JSON file for future use
import json

with open(os.path.join(PROCESSED_DATA_PATH, "label_mappings.json"), "w") as f:
    json.dump(label_mappings, f)

# ================================================================
# Data Visualization (Optional)
# ================================================================
# You can create plots to visualize the distribution of classes if needed.

# ================================================================
# Custom Dataset Class
# ================================================================
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_df, transform=None):
        """
        Custom dataset for plant disease classification that handles
        - Crop Type Classification
        - Disease Detection (Healthy vs Diseased)
        - Disease Type Classification

        Args:
            data_df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # Get image path and labels
        img_path = self.data_df.loc[idx, "image_path"]
        plant_label = self.data_df.loc[idx, "plant_label"]
        disease_label = self.data_df.loc[idx, "disease_label"]
        healthy_label = self.data_df.loc[idx, "healthy_label"]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Create label dictionary
        labels = {
            "crop_type": torch.tensor(plant_label, dtype=torch.long),
            "disease": torch.tensor(disease_label, dtype=torch.long),
            "healthy": torch.tensor(healthy_label, dtype=torch.float32),
        }

        return image, labels

# ================================================================
# Data Transformations
# ================================================================
# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ================================================================
# Creating Dataset Instances
# ================================================================
# Split data into training and validation DataFrames
train_df = data_df[data_df["split"] == "train"].reset_index(drop=True)
valid_df = data_df[data_df["split"] == "valid"].reset_index(drop=True)

# Create dataset instances
train_dataset = PlantDiseaseDataset(train_df, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_df, transform=transform)

# ================================================================
# Creating Data Loaders
# ================================================================
# Parameters
BATCH_SIZE = 32
NUM_WORKERS = 8  # Adjust based on your system

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

# ================================================================
# Testing the Dataset and DataLoader
# ================================================================
# Fetch a batch of data to test
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"\nBatch of images shape: {images.shape}")
print(f"Batch of labels: {labels}")

# ================================================================
# Saving the Prepared Dataset (Optional)
# ================================================================
# If you wish to save the prepared dataset indices for reproducibility
train_indices = train_df.index.tolist()
valid_indices = valid_df.index.tolist()

np.save(os.path.join(PROCESSED_DATA_PATH, "train_indices.npy"), train_indices)
np.save(os.path.join(PROCESSED_DATA_PATH, "valid_indices.npy"), valid_indices)

# ================================================================
# Summary
# ================================================================
print("\nDataset Preparation Complete!")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of crop types: {len(label_mappings['plant_to_idx'])}")
print(f"Number of diseases: {len(label_mappings['disease_to_idx'])}")
