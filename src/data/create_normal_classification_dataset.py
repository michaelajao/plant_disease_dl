# create_normal_classification_dataset.py

# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import random
import shutil
import argparse
from tqdm import tqdm

# ================================================================
# Configuration and Settings
# ================================================================
# Default dataset paths
RAW_DATA_PATH = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
PROCESSED_DATA_PATH = "../../data/processed/normal_classification/"

# Ensure the processed data directory exists
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# ================================================================
# Function to Create Dataset
# ================================================================
def create_normal_classification_dataset(
    num_classes=None, train_split=0.8, seed=42
):
    """
    Creates a normal classification dataset with specified number of classes and split.

    Args:
        num_classes (int): Number of classes to include. If None, include all classes.
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

    # Select subset of classes if specified
    if num_classes is not None and num_classes < len(all_classes):
        selected_classes = random.sample(all_classes, num_classes)
    else:
        selected_classes = all_classes

    print(f"Selected {len(selected_classes)} classes for the dataset.")

    # Create processed data directories
    processed_train_dir = os.path.join(PROCESSED_DATA_PATH, "train")
    processed_valid_dir = os.path.join(PROCESSED_DATA_PATH, "valid")
    if not os.path.exists(processed_train_dir):
        os.makedirs(processed_train_dir)
    if not os.path.exists(processed_valid_dir):
        os.makedirs(processed_valid_dir)

    # Copy and split data
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

        # Combine and shuffle images
        all_images = train_images + valid_images
        random.shuffle(all_images)

        # Split images into training and validation sets
        split_index = int(len(all_images) * train_split)
        train_images = all_images[:split_index]
        valid_images = all_images[split_index:]

        # Create class directories in processed data
        proc_class_train_dir = os.path.join(processed_train_dir, class_name)
        proc_class_valid_dir = os.path.join(processed_valid_dir, class_name)
        if not os.path.exists(proc_class_train_dir):
            os.makedirs(proc_class_train_dir)
        if not os.path.exists(proc_class_valid_dir):
            os.makedirs(proc_class_valid_dir)

        # Copy images to processed data directories
        for img_path in train_images:
            shutil.copy(img_path, proc_class_train_dir)
        for img_path in valid_images:
            shutil.copy(img_path, proc_class_valid_dir)

    print("Dataset creation completed.")

# ================================================================
# Main Function
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Normal Classification Dataset")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes to include (default: all classes)",
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

    create_normal_classification_dataset(
        num_classes=args.num_classes, train_split=args.train_split, seed=args.seed
    )
