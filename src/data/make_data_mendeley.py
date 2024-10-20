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

# Ensure the helper functions and settings are imported
# Adjust the path as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds  # Ensure this function is defined appropriately

# ================================================================
# Configuration and Settings
# ================================================================
# Hyperparameters and configuration setup
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Default Directory paths (can be overridden via command-line arguments)
DEFAULT_RAW_DATA_PATH = "../../data/raw/plant leaf disease dataset"
DEFAULT_PROCESSED_DATA_PATH = "../../data/processed/plant_leaf_disease_dataset/"

# ================================================================
# Function to Create Multitask Classification Dataset
# ================================================================
def create_multitask_classification_dataset(
    source_dir,
    processed_dir,
    num_plants=None,
    num_diseases=None,
    train_split=0.8,
    seed=42
):
    """
    Creates a multitask classification dataset with specified number of plants and diseases.

    Args:
        source_dir (str): Path to the raw dataset directory.
        processed_dir (str): Path to save the processed dataset.
        num_plants (int): Number of plant types to include.
        num_diseases (int): Number of diseases to include.
        train_split (float): Fraction of data to use for training.
        seed (int): Random seed for reproducibility.
    """
    print("\n=== Creating Multitask Classification Dataset ===")
    random.seed(seed)

    # Get list of all classes (subdirectories)
    all_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    all_classes.sort()

    # Extract plant types and diseases
    plant_types = set()
    diseases = set()
    for class_name in all_classes:
        try:
            plant, disease = class_name.split("___")
            plant_types.add(plant)
            diseases.add(disease)
        except ValueError:
            print(f"Skipping invalid class directory name: {class_name}")
            continue

    plant_types = sorted(list(plant_types))
    diseases = sorted(list(diseases))

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
        try:
            plant, disease = class_name.split("___")
            if plant in selected_plants and disease in selected_diseases:
                selected_classes.append(class_name)
        except ValueError:
            continue  # Skip invalid class names

    print(f"Total classes selected for multitask: {len(selected_classes)}")

    if not selected_classes:
        print("No classes selected for multitask dataset. Skipping.")
        return

    # Create processed data directories for multitask
    multitask_processed_dir = os.path.join(processed_dir, "multitask")
    processed_train_dir = os.path.join(multitask_processed_dir, "train")
    processed_valid_dir = os.path.join(multitask_processed_dir, "valid")
    os.makedirs(processed_train_dir, exist_ok=True)
    os.makedirs(processed_valid_dir, exist_ok=True)

    # Prepare labels mapping
    plant_to_idx = {plant: idx for idx, plant in enumerate(selected_plants)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(selected_diseases)}
    healthy_label = {"Healthy": 1, "Diseased": 0}

    labels_mapping = {
        "plant_to_idx": plant_to_idx,
        "disease_to_idx": disease_to_idx,
        "healthy_label": healthy_label,
    }

    # Save labels mapping for multitask
    multitask_mapping_path = os.path.join(multitask_processed_dir, "labels_mapping_multitask.json")
    with open(multitask_mapping_path, "w") as f:
        json.dump(labels_mapping, f, indent=4)

    # Initialize list to hold dataset records
    data_records = []

    # Iterate through each selected class
    for class_name in tqdm(selected_classes, desc="Processing Multitask Classes"):
        class_dir = os.path.join(source_dir, class_name)

        # Get all image files in the class directory
        all_images = [
            os.path.join(class_dir, img)
            for img in os.listdir(class_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not all_images:
            print(f"No images found in class directory: {class_dir}. Skipping.")
            continue

        # Shuffle images to ensure random split
        random.shuffle(all_images)

        # Split into training and validation sets
        split_index = int(len(all_images) * train_split)
        train_images = all_images[:split_index]
        valid_images = all_images[split_index:]

        # Extract labels
        try:
            plant, disease = class_name.split("___")
        except ValueError:
            print(f"Invalid class name format: {class_name}. Skipping.")
            continue

        plant_idx = plant_to_idx[plant]
        disease_idx = disease_to_idx[disease]
        healthy = 1 if disease.lower() == "healthy" else 0

        # Copy training images
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_train_dir, img_name)

            # Handle potential filename conflicts
            unique_dest_path = get_unique_filepath(dest_path)

            shutil.copy(img_path, unique_dest_path)

            data_records.append(
                {
                    "image": unique_dest_path,
                    "plant": plant_idx,
                    "disease": disease_idx,
                    "healthy": healthy,
                    "split": "train",
                }
            )

        # Copy validation images
        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_valid_dir, img_name)

            # Handle potential filename conflicts
            unique_dest_path = get_unique_filepath(dest_path)

            shutil.copy(img_path, unique_dest_path)

            data_records.append(
                {
                    "image": unique_dest_path,
                    "plant": plant_idx,
                    "disease": disease_idx,
                    "healthy": healthy,
                    "split": "valid",
                }
            )

    # Save the multitask dataset records to a CSV file
    multitask_df = pd.DataFrame(data_records)
    multitask_csv_path = os.path.join(multitask_processed_dir, "dataset_multitask.csv")
    multitask_df.to_csv(multitask_csv_path, index=False)

    print("Multitask dataset creation completed.")

# ================================================================
# Function to Create Single-Task Classification Dataset
# ================================================================
def create_single_task_classification_dataset(
    source_dir,
    processed_dir,
    task="disease",  # or "plant"
    num_classes=None,
    train_split=0.8,
    seed=42
):
    """
    Creates a single-task classification dataset (either plant or disease).

    Args:
        source_dir (str): Path to the raw dataset directory.
        processed_dir (str): Path to save the processed dataset.
        task (str): The task to perform, either "disease" or "plant".
        num_classes (int): Number of classes to include. If None, include all.
        train_split (float): Fraction of data to use for training.
        seed (int): Random seed for reproducibility.
    """
    print(f"\n=== Creating Single-Task Classification Dataset: {task.capitalize()} ===")
    if task not in ["disease", "plant"]:
        raise ValueError('Task must be either "disease" or "plant".')

    random.seed(seed)

    # Get list of all classes (subdirectories)
    all_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    all_classes.sort()

    # Extract labels based on the task
    labels_set = set()
    class_to_label = {}
    for class_name in all_classes:
        try:
            plant, disease = class_name.split("___")
            label = disease if task == "disease" else plant
            labels_set.add(label)
            class_to_label[class_name] = label
        except ValueError:
            print(f"Skipping invalid class directory name: {class_name}")
            continue

    labels = sorted(list(labels_set))

    # Select subset of classes if specified
    if num_classes is not None and num_classes < len(labels):
        selected_labels = random.sample(labels, num_classes)
    else:
        selected_labels = labels

    print(f"Selected {len(selected_labels)} {task} classes.")

    # Filter classes based on selected labels
    selected_classes = [
        class_name for class_name, label in class_to_label.items() if label in selected_labels
    ]

    print(f"Total classes selected for single-task ({task}): {len(selected_classes)}")

    if not selected_classes:
        print(f"No classes selected for single-task ({task}) dataset. Skipping.")
        return

    # Create processed data directories for single-task
    single_task_processed_dir = os.path.join(processed_dir, f"single_task_{task}")
    processed_train_dir = os.path.join(single_task_processed_dir, "train")
    processed_valid_dir = os.path.join(single_task_processed_dir, "valid")
    os.makedirs(processed_train_dir, exist_ok=True)
    os.makedirs(processed_valid_dir, exist_ok=True)

    # Prepare labels mapping
    label_to_idx = {label: idx for idx, label in enumerate(selected_labels)}
    labels_mapping = {
        f"{task}_to_idx": label_to_idx
    }

    # Save labels mapping for single-task
    mapping_filename = f"labels_mapping_single_task_{task}.json"
    mapping_path = os.path.join(single_task_processed_dir, mapping_filename)
    with open(mapping_path, "w") as f:
        json.dump(labels_mapping, f, indent=4)

    # Initialize list to hold dataset records
    data_records = []

    # Iterate through each selected class
    for class_name in tqdm(selected_classes, desc=f"Processing Single-Task ({task.capitalize()}) Classes"):
        class_dir = os.path.join(source_dir, class_name)

        # Get all image files in the class directory
        all_images = [
            os.path.join(class_dir, img)
            for img in os.listdir(class_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not all_images:
            print(f"No images found in class directory: {class_dir}. Skipping.")
            continue

        # Shuffle images to ensure random split
        random.shuffle(all_images)

        # Split into training and validation sets
        split_index = int(len(all_images) * train_split)
        train_images = all_images[:split_index]
        valid_images = all_images[split_index:]

        # Extract label
        try:
            plant, disease = class_name.split("___")
            label = disease if task == "disease" else plant
        except ValueError:
            print(f"Invalid class name format: {class_name}. Skipping.")
            continue

        label_idx = label_to_idx[label]

        # Copy training images
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_train_dir, img_name)

            # Handle potential filename conflicts
            unique_dest_path = get_unique_filepath(dest_path)

            shutil.copy(img_path, unique_dest_path)

            data_records.append(
                {
                    "image": unique_dest_path,
                    "label": label_idx,
                    "split": "train",
                }
            )

        # Copy validation images
        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_valid_dir, img_name)

            # Handle potential filename conflicts
            unique_dest_path = get_unique_filepath(dest_path)

            shutil.copy(img_path, unique_dest_path)

            data_records.append(
                {
                    "image": unique_dest_path,
                    "label": label_idx,
                    "split": "valid",
                }
            )

    # Save the single-task dataset records to a CSV file
    single_task_df = pd.DataFrame(data_records)
    single_task_csv_path = os.path.join(single_task_processed_dir, f"dataset_single_task_{task}.csv")
    single_task_df.to_csv(single_task_csv_path, index=False)

    print(f"Single-task ({task}) dataset creation completed.")

# ================================================================
# Utility Function to Handle Filename Conflicts
# ================================================================
def get_unique_filepath(dest_path):
    """
    Generates a unique file path by appending a counter if the file already exists.

    Args:
        dest_path (str): The intended destination file path.

    Returns:
        str: A unique file path.
    """
    unique_dest_path = dest_path
    count = 1
    while os.path.exists(unique_dest_path):
        name, ext = os.path.splitext(os.path.basename(dest_path))
        unique_dest_path = os.path.join(os.path.dirname(dest_path), f"{name}_{count}{ext}")
        count += 1
    return unique_dest_path

# ================================================================
# Main Function
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Classification Datasets for Plant Leaf Diseases"
    )
    parser.add_argument(
        "--num_plants",
        type=int,
        default=None,
        help="Number of plant types to include for multitask dataset (default: all plants)",
    )
    parser.add_argument(
        "--num_diseases",
        type=int,
        default=None,
        help="Number of disease types to include for multitask dataset (default: all diseases)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes to include for single-task classification (default: all classes)",
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
    parser.add_argument(
        "--source_dir",
        type=str,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the raw dataset directory",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path to save the processed dataset",
    )
    parser.add_argument(
        "--create_multitask",
        action='store_true',
        help="Flag to create multitask classification dataset",
    )
    parser.add_argument(
        "--create_single_task_disease",
        action='store_true',
        help="Flag to create single-task disease classification dataset",
    )
    parser.add_argument(
        "--create_single_task_plant",
        action='store_true',
        help="Flag to create single-task plant classification dataset",
    )

    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seeds(args.seed)

    # Check which datasets to create
    datasets_to_create = []
    if args.create_multitask:
        datasets_to_create.append("multitask")
    if args.create_single_task_disease:
        datasets_to_create.append("single_task_disease")
    if args.create_single_task_plant:
        datasets_to_create.append("single_task_plant")

    if not datasets_to_create:
        print("No dataset creation flags provided. Please specify at least one of the following flags:")
        print("--create_multitask, --create_single_task_disease, --create_single_task_plant")
        sys.exit(1)

    # Create each requested dataset
    for dataset in datasets_to_create:
        if dataset == "multitask":
            create_multitask_classification_dataset(
                source_dir=args.source_dir,
                processed_dir=args.processed_dir,
                num_plants=args.num_plants,
                num_diseases=args.num_diseases,
                train_split=args.train_split,
                seed=args.seed,
            )
        elif dataset == "single_task_disease":
            create_single_task_classification_dataset(
                source_dir=args.source_dir,
                processed_dir=args.processed_dir,
                task="disease",
                num_classes=args.num_classes,
                train_split=args.train_split,
                seed=args.seed,
            )
        elif dataset == "single_task_plant":
            create_single_task_classification_dataset(
                source_dir=args.source_dir,
                processed_dir=args.processed_dir,
                task="plant",
                num_classes=args.num_classes,
                train_split=args.train_split,
                seed=args.seed,
            )
