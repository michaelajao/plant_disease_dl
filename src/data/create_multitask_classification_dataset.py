# create_multitask_classification_dataset.py

# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import random
import shutil
import argparse
import json
from tqdm import tqdm
import pandas as pd

# ================================================================
# Configuration and Settings
# ================================================================
# Default dataset paths
RAW_DATA_PATH = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
PROCESSED_DATA_PATH = "../../data/processed/multitask_classification/"

# Ensure the processed data directory exists
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# ================================================================
# Function to Create Dataset
# ================================================================
def create_multitask_classification_dataset(
    num_plants=None, num_diseases=None, train_split=0.8, seed=42
):
    """
    Creates a multitask classification dataset with specified number of plants and diseases.

    Args:
        num_plants (int): Number of plant types to include.
        num_diseases (int): Number of diseases to include.
        train_split (float): Fraction of data to use for training.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    # Directories
    train_dir = os.path.join(RAW_DATA_PATH, "train")
    valid_dir = os.path.join(RAW_DATA_PATH, "valid")

    # Get list of all classes (disease folders)
    all_classes = os.listdir(train_dir)
    all_classes.sort()

    # Extract plant types and diseases
    plant_types = set()
    diseases = set()
    for class_name in all_classes:
        plant, disease = class_name.split("___")
        plant_types.add(plant)
        diseases.add(disease)

    plant_types = list(plant_types)
    diseases = list(diseases)
    plant_types.sort()
    diseases.sort()

    # Select subset of plants and diseases if specified
    if num_plants is not None and num_plants < len(plant_types):
        selected_plants = random.sample(plant_types, num_plants)
    else:
        selected_plants = plant_types

    if num_diseases is not None and num_diseases < len(diseases):
        selected_diseases = random.sample(diseases, num_diseases)
    else:
        selected_diseases = diseases

    print(f"Selected {len(selected_plants)} plants and {len(selected_diseases)} diseases.")

    # Filter classes based on selected plants and diseases
    selected_classes = []
    for class_name in all_classes:
        plant, disease = class_name.split("___")
        if plant in selected_plants and disease in selected_diseases:
            selected_classes.append(class_name)

    print(f"Total classes selected: {len(selected_classes)}")

    # Create processed data directories
    processed_train_dir = os.path.join(PROCESSED_DATA_PATH, "train")
    processed_valid_dir = os.path.join(PROCESSED_DATA_PATH, "valid")
    if not os.path.exists(processed_train_dir):
        os.makedirs(processed_train_dir)
    if not os.path.exists(processed_valid_dir):
        os.makedirs(processed_valid_dir)

    # Prepare labels mapping
    plant_to_idx = {plant: idx for idx, plant in enumerate(selected_plants)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(selected_diseases)}
    healthy_label = {"Healthy": 1, "Diseased": 0}

    labels_mapping = {
        "plant_to_idx": plant_to_idx,
        "disease_to_idx": disease_to_idx,
        "healthy_label": healthy_label,
    }

    # Save labels mapping for future use
    with open(os.path.join(PROCESSED_DATA_PATH, "labels_mapping.json"), "w") as f:
        json.dump(labels_mapping, f)

    # Copy and split data
    data_records = []
    for class_name in tqdm(selected_classes, desc="Processing Classes"):
        class_train_dir = os.path.join(train_dir, class_name)
        class_valid_dir = os.path.join(valid_dir, class_name)

        # Get all images for this class
        train_images = [
            os.path.join(class_train_dir, img)
            for img in os.listdir(class_train_dir)
            if img.endswith((".jpg", ".png"))
        ]
        valid_images = [
            os.path.join(class_valid_dir, img)
            for img in os.listdir(class_valid_dir)
            if img.endswith((".jpg", ".png"))
        ]

        # Combine images
        all_images = train_images + valid_images

        # Create labels for the images
        plant, disease = class_name.split("___")
        plant_idx = plant_to_idx[plant]
        disease_idx = disease_to_idx[disease]
        healthy = 1 if disease == "Healthy" else 0

        # Split images into training and validation sets
        random.shuffle(all_images)
        split_index = int(len(all_images) * train_split)
        train_images = all_images[:split_index]
        valid_images = all_images[split_index:]

        # Copy images and record data
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_train_dir, img_name)
            shutil.copy(img_path, dest_path)
            data_records.append(
                {
                    "image": dest_path,
                    "plant": plant_idx,
                    "disease": disease_idx,
                    "healthy": healthy,
                    "split": "train",
                }
            )

        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_valid_dir, img_name)
            shutil.copy(img_path, dest_path)
            data_records.append(
                {
                    "image": dest_path,
                    "plant": plant_idx,
                    "disease": disease_idx,
                    "healthy": healthy,
                    "split": "valid",
                }
            )

    # Save the dataset records to a CSV file
    data_df = pd.DataFrame(data_records)
    data_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "dataset.csv"), index=False)

    print("Dataset creation completed.")

# ================================================================
# Main Function
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Multi-Task Multi-Class Classification Dataset"
    )
    parser.add_argument(
        "--num_plants",
        type=int,
        default=None,
        help="Number of plant types to include (default: all plants)",
    )
    parser.add_argument(
        "--num_diseases",
        type=int,
        default=None,
        help="Number of diseases to include (default: all diseases)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    create_multitask_classification_dataset(
        num_plants=args.num_plants,
        num_diseases=args.num_diseases,
        train_split=args.train_split,
        seed=args.seed,
    )
