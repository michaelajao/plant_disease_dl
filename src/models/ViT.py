# src/models/ViT_refactored.py

# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# tqdm for progress bars
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Ensure the helper functions and settings are imported
# (Assuming helper_functions.py exists and contains set_seeds)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds  # Removed accuracy_fn as it's unused
from helper_functions import *

# ================================================================
# Setup Logging
# ================================================================
logging.basicConfig(
    filename='training_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# ================================================================
# Configuration and Settings
# ================================================================

# Set seeds for reproducibility
set_seeds(42)

# Hyperparameters
BATCH_SIZE = 32          # Adjust based on GPU memory
LEARNING_RATE = 1e-4     # Lower learning rate for training from scratch
NUM_EPOCHS = 50          # Increased epochs for better training
HEIGHT, WIDTH = 224, 224 # Image dimensions

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 10  # Increased patience for early stopping

# ================================================================
# Device Configuration
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs")
print(f"Using device: {device}")

# ================================================================
# Directory Setup
# ================================================================

# Define project root (assuming this script is in src/models/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define directories for data and models
data_path = os.path.join(
    project_root,
    "data",
    "processed",
    "plant_leaf_disease_dataset",
    "single_task_disease",
)
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define output directories for results, figures, and models
output_dirs = [
    os.path.join(project_root, "reports", "results"),
    os.path.join(project_root, "reports", "figures"),
    os.path.join(project_root, "models"),
]

# Create output directories if they don't exist
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Function to list directory contents
def list_directory_contents(directory, num_items=10):
    if os.path.exists(directory):
        contents = os.listdir(directory)
        print(
            f"Contents of {directory} ({len(contents)} items): {contents[:num_items]}..."
        )
    else:
        print(f"Directory does not exist: {directory}")

# Verify directories and list contents
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Validation directory exists: {os.path.exists(valid_dir)}")
list_directory_contents(train_dir, num_items=10)
list_directory_contents(valid_dir, num_items=10)

# ================================================================
# Load Label Mappings
# ================================================================

# Path to label mapping JSON
labels_mapping_path = os.path.join(data_path, "labels_mapping_single_task_disease.json")

# Load the label mapping
if os.path.exists(labels_mapping_path):
    with open(labels_mapping_path, "r") as f:
        labels_mapping = json.load(f)

    disease_to_idx = labels_mapping.get("disease_to_idx", {})
    if not disease_to_idx:
        print("Error: 'disease_to_idx' mapping not found in the JSON file.")
        sys.exit(1)

    idx_to_disease = {v: k for k, v in disease_to_idx.items()}
    print(f"Disease to Index Mapping: {disease_to_idx}")
    print(f"Index to Disease Mapping: {idx_to_disease}")
else:
    print(f"Warning: Label mapping file not found at {labels_mapping_path}. Exiting.")
    sys.exit(1)  # Exit, as proper label mapping is essential

# ================================================================
# Define Minority Classes
# ================================================================

# Define minority classes based on training label counts
# You can adjust the threshold as needed
minority_threshold = 1000  # Classes with fewer than 1000 samples are considered minority

# Compute label counts from the training set
train_split_csv = os.path.join(data_path, "train_split.csv")
if os.path.exists(train_split_csv):
    train_df = pd.read_csv(train_split_csv)
    train_label_counts = train_df['label'].value_counts().sort_index()
else:
    print(f"Error: Training split CSV not found at {train_split_csv}. Exiting.")
    sys.exit(1)

minority_classes = train_label_counts[train_label_counts < minority_threshold].index.tolist()

print(f"\nIdentified Minority Classes (count < {minority_threshold}):")
for cls in minority_classes:
    print(f"Class {cls} ({idx_to_disease.get(cls, 'Unknown')}) with {train_label_counts[cls]} samples")

# ================================================================
# Custom Dataset Class with Class-Specific Augmentations
# ================================================================

class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform_major=None, transform_minority=None,
                 minority_classes=None, image_col='image', label_col='label'):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            images_dir (str): Directory with all the images.
            transform_major (callable, optional): Transformations for majority classes.
            transform_minority (callable, optional): Transformations for minority classes.
            minority_classes (list, optional): List of minority class indices.
            image_col (str): Column name for image filenames in the CSV.
            label_col (str): Column name for labels in the CSV.
        """
        self.annotations = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform_major = transform_major
        self.transform_minority = transform_minority
        self.minority_classes = minority_classes if minority_classes else []
        self.image_col = image_col
        self.label_col = label_col

        # Verify required columns
        required_columns = [image_col, label_col]
        for col in required_columns:
            if col not in self.annotations.columns:
                raise ValueError(f"Missing required column '{col}' in CSV file.")

        # Ensure labels are integers
        if not pd.api.types.is_integer_dtype(self.annotations[self.label_col]):
            try:
                self.annotations[self.label_col] = self.annotations[self.label_col].astype(int)
                print(f"Converted labels in {csv_file} to integers.")
            except ValueError:
                print(f"Error: Labels in {csv_file} cannot be converted to integers.")
                self.annotations[self.label_col] = -1  # Assign invalid label

        # Debug: Print unique labels after conversion
        unique_labels = self.annotations[self.label_col].unique()
        print(f"Unique labels after conversion in {csv_file}: {unique_labels}")

        # Check labels are within [0, num_classes - 1]
        num_classes = len(disease_to_idx)
        valid_labels = self.annotations[self.label_col].between(0, num_classes - 1)
        invalid_count = len(self.annotations) - valid_labels.sum()
        if invalid_count > 0:
            print(f"Found {invalid_count} samples with invalid labels in {csv_file}. These will be skipped.")
            self.annotations = self.annotations[valid_labels].reset_index(drop=True)

        # Final count
        print(f"Number of samples after filtering in {csv_file}: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image filename and label
        img_name_full = self.annotations.iloc[idx][self.image_col]
        label_idx = self.annotations.iloc[idx][self.label_col]

        # Extract only the basename to avoid path duplication
        img_name = os.path.basename(img_name_full)

        # Full path to the image
        img_path = os.path.join(self.images_dir, img_name)

        # Open image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new("RGB", (HEIGHT, WIDTH), (0, 0, 0))

        # Apply class-specific transformations
        if label_idx in self.minority_classes and self.transform_minority:
            image = self.transform_minority(image)
        elif self.transform_major:
            image = self.transform_major(image)

        return image, label_idx

# ================================================================
# Split Dataset into Training, Validation, and Test Sets
# ================================================================

# Paths to CSV files
full_csv = os.path.join(data_path, "dataset_single_task_disease.csv")
train_split_csv = os.path.join(data_path, "train_split.csv")
valid_split_csv = os.path.join(data_path, "valid_split.csv")
test_split_csv = os.path.join(data_path, "test_split.csv")  # New test split

# Read the full CSV
if os.path.exists(full_csv):
    full_df = pd.read_csv(full_csv)
    print(f"\nFull dataset contains {len(full_df)} samples.")
else:
    print(f"Error: Full dataset CSV not found at {full_csv}. Exiting.")
    sys.exit(1)

# Check if 'split' column exists
if "split" in full_df.columns:
    train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
    valid_df = full_df[full_df["split"] == "valid"].reset_index(drop=True)
    test_df = full_df[full_df["split"] == "test"].reset_index(drop=True) if 'test' in full_df['split'].unique() else pd.DataFrame()
    print("Dataset split based on 'split' column.")
else:
    # If no 'split' column, perform an 80-10-10 split
    from sklearn.model_selection import train_test_split

    if 'label' not in full_df.columns:
        raise ValueError("CSV file must contain a 'label' column for stratified splitting.")

    train_df, temp_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["label"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )
    print("Dataset split into 80% training, 10% validation, and 10% testing.")

# Save the split CSVs
train_df.to_csv(train_split_csv, index=False)
valid_df.to_csv(valid_split_csv, index=False)
if not test_df.empty:
    test_df.to_csv(test_split_csv, index=False)
    print(f"Saved test split to {test_split_csv} with {len(test_df)} samples.")
print(f"Saved training split to {train_split_csv} with {len(train_df)} samples.")
print(f"Saved validation split to {valid_split_csv} with {len(valid_df)} samples.")

# ================================================================
# Data Transforms with Class-Specific Augmentations (Reintroduced Normalization)
# ================================================================

# Define transforms for majority classes
transform_major = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
        std=[0.229, 0.224, 0.225]     # Std for ImageNet
    ),
])

# Define transforms for minority classes with additional augmentations
transform_minority = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Additional flip
    transforms.RandomRotation(30),    # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
        std=[0.229, 0.224, 0.225]     # Std for ImageNet
    ),
])

# ================================================================
# Initialize Datasets and DataLoaders (Using Pre-trained ViT)
# ================================================================

# Initialize training, validation, and test datasets with class-specific transforms
train_dataset = PlantDiseaseDataset(
    csv_file=train_split_csv,
    images_dir=train_dir,
    transform_major=transform_major,
    transform_minority=transform_minority,
    minority_classes=minority_classes,
    image_col='image',  # Ensure this matches the actual column name in your CSV
    label_col='label'   # Ensure this matches the actual column name in your CSV
)

valid_dataset = PlantDiseaseDataset(
    csv_file=valid_split_csv,
    images_dir=valid_dir,
    transform_major=transform_major,  # Validation should not have augmentation
    transform_minority=None,          # No augmentation for validation
    minority_classes=[],              # No augmentation needed
    image_col='image',
    label_col='label'
)

# Initialize test dataset if available
if os.path.exists(test_split_csv):
    test_dataset = PlantDiseaseDataset(
        csv_file=test_split_csv,
        images_dir=valid_dir,  # Assuming test images are in the same directory as validation
        transform_major=transform_major,  # No augmentation
        transform_minority=None,
        minority_classes=[],
        image_col='image',
        label_col='label'
    )
else:
    test_dataset = None
    print("No test split found. Skipping test dataset initialization.")

# Create DataLoaders (Using shuffle=True for training)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # Enabled shuffling for better generalization
    num_workers=4, 
    pin_memory=True if torch.cuda.is_available() else False
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True if torch.cuda.is_available() else False
) if test_dataset else None

# Display dataset information
print(f"\nNumber of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
if test_dataset:
    print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of classes: {len(disease_to_idx)}")
print(f"Classes: {list(disease_to_idx.keys())}")

# Test fetching a single sample
if len(train_dataset) > 0:
    sample_image, sample_label = train_dataset[0]
    print(f"\nSample Image Shape: {sample_image.shape}")
    print(f"Sample Label Index: {sample_label}")
    print(f"Sample Label Name: {idx_to_disease.get(sample_label, 'Unknown')}")
else:
    print("\nTraining dataset is empty. Please check your dataset and label mappings.")

# ================================================================
# Data Visualization: Plotting Label Distribution (Removed Initial Visualization)
# ================================================================
def plot_label_distribution_pandas(csv_path, idx_to_disease, dataset_name="Training"):
    """
    Plot the distribution of labels using Pandas' built-in plotting.
    
    Args:
        csv_path (str): Path to the CSV file.
        idx_to_disease (dict): Mapping from index to disease name.
        dataset_name (str): Name of the dataset split (for the plot title).
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Cannot plot label distribution.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nPlotting label distribution using all {len(df)} samples from the {dataset_name} dataset.")

    # Verify 'label' column exists
    if 'label' not in df.columns:
        print(f"'label' column not found in {csv_path}. Cannot plot label distribution.")
        return

    # Compute label counts
    label_counts = df['label'].value_counts().sort_index()

    # Map label indices to disease names
    label_counts.index = label_counts.index.map(idx_to_disease)

    # Handle any unmapped labels
    label_counts = label_counts.fillna("Unknown")

    # Debug: Check label counts
    print(f"Label counts:\n{label_counts}")

    # Plot using Pandas
    plt.figure(figsize=(12, 8))
    label_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Label Distribution in {dataset_name} Dataset")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Save the plot in the figures directory
    plt.savefig(os.path.join(output_dirs[1], f"label_distribution_{dataset_name.lower()}.pdf"))
    plt.show()

# Plot label distribution for training and validation sets
plot_label_distribution_pandas(train_split_csv, idx_to_disease, "Training")
plot_label_distribution_pandas(valid_split_csv, idx_to_disease, "Validation")
if test_split_csv and os.path.exists(test_split_csv):
    plot_label_distribution_pandas(test_split_csv, idx_to_disease, "Test")

# ================================================================
# Visualization Functions to Verify Augmentation (Removed Initial Visualization)
# ================================================================

def plot_augmented_samples(dataset, idx_to_disease, num_samples=5):
    """
    Plot multiple augmented samples from each minority class.
    
    Args:
        dataset (Dataset): The dataset to sample from.
        idx_to_disease (dict): Mapping from label index to disease name.
        num_samples (int): Number of samples to plot per class.
    """
    # Create a dictionary to hold samples for each minority class
    samples_per_class = {cls: [] for cls in dataset.minority_classes}

    # Iterate through the dataset and collect samples
    for img, label in dataset:
        if label.item() in dataset.minority_classes:
            samples_per_class[label.item()].append(img)
            if all(len(imgs) >= num_samples for imgs in samples_per_class.values()):
                break  # Stop once we have enough samples for each class

    # Plot the samples
    for cls, images in samples_per_class.items():
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Augmented Samples for Class: {idx_to_disease.get(cls, 'Unknown')}", fontsize=16)
        for i in range(num_samples):
            if i < len(images):
                image = images[i].cpu().numpy().transpose((1, 2, 0))
                # Unnormalize for display
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
                plt.subplot(1, num_samples, i+1)
                plt.imshow(image)
                plt.axis('off')
        # Save the plot in the figures directory
        plt.savefig(os.path.join(output_dirs[1], f"augmented_samples_class_{cls}.pdf"))
        plt.show()

def plot_original_and_augmented(dataset, idx_to_disease, class_index, num_pairs=3):
    """
    Plot original and augmented image pairs from a specified class.
    
    Args:
        dataset (Dataset): The dataset to sample from.
        idx_to_disease (dict): Mapping from label index to disease name.
        class_index (int): The label index of the class to visualize.
        num_pairs (int): Number of image pairs to plot.
    """
    plt.figure(figsize=(10, num_pairs * 4))
    plt.suptitle(f"Original vs. Augmented Samples for Class: {idx_to_disease.get(class_index, 'Unknown')}", fontsize=16)
    
    count = 0
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label.item() == class_index:
            # Original Image (without augmentation)
            original_transform = transforms.Compose([
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
                    std=[0.229, 0.224, 0.225]     # Std for ImageNet
                ),
            ])
            img_name = dataset.annotations.iloc[idx]['image']
            img_path = os.path.join(dataset.images_dir, os.path.basename(img_name))
            try:
                original_image = Image.open(img_path).convert("RGB")
                original_image = original_transform(original_image).numpy().transpose((1, 2, 0))
                # Unnormalize for display
                original_image = original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                original_image = np.clip(original_image, 0, 1)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # Augmented Image
            augmented_image = img.numpy().transpose((1, 2, 0))
            # Unnormalize for display
            augmented_image = augmented_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            augmented_image = np.clip(augmented_image, 0, 1)
            
            # Plot Original
            plt.subplot(num_pairs, 2, 2*count + 1)
            plt.imshow(original_image)
            plt.title("Original")
            plt.axis('off')
            
            # Plot Augmented
            plt.subplot(num_pairs, 2, 2*count + 2)
            plt.imshow(augmented_image)
            plt.title("Augmented")
            plt.axis('off')
            
            count += 1
            if count >= num_pairs:
                break
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot in the figures directory
    plt.savefig(os.path.join(output_dirs[1], f"original_vs_augmented_class_{class_index}.pdf"))
    plt.show()

def plot_random_image_from_loader(dataloader, idx_to_disease, output_dirs):
    """
    Plot a random image from the dataloader with its label.

    Args:
        dataloader (DataLoader): DataLoader to fetch the image from.
        idx_to_disease (dict): Mapping from index to disease name.
        output_dirs (list): List of output directories for saving plots.
    """
    if len(dataloader) == 0:
        print("Dataloader is empty. Cannot plot image.")
        return

    # Get a single batch of data
    try:
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
    except StopIteration:
        print("Dataloader has no data.")
        return

    if len(images) == 0:
        print("No images in the batch to plot.")
        return

    # Randomly select an index from the batch
    random_idx = random.randint(0, len(images) - 1)

    # Select the random image and label
    image = images[random_idx]
    label = labels[random_idx]

    # Convert the image to numpy array
    image = image.cpu().numpy().transpose((1, 2, 0))
    # Unnormalize for display
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    label_name = idx_to_disease.get(label.item(), "Unknown")
    plt.title(f"Label: {label_name}")
    plt.axis('off')
    # Save the plot in the figures directory
    plt.savefig(os.path.join(output_dirs[1], "random_image_from_loader.pdf"))
    plt.show()

# ================================================================
# Visualization of Training Data
# ================================================================

# Uncomment the following lines if you wish to visualize augmented samples
plot_augmented_samples(train_dataset, idx_to_disease, num_samples=5)

# Uncomment the following lines to plot original vs augmented images for a specific class
# Choose a minority class to visualize, e.g., the first minority class
if minority_classes:
    visualize_class_index = minority_classes[0]
    plot_original_and_augmented(train_dataset, idx_to_disease, visualize_class_index, num_pairs=3)

# Plot a random image from the train_loader
if len(train_dataset) > 0:
    plot_random_image_from_loader(train_loader, idx_to_disease, output_dirs)
else:
    print("Cannot plot image: Training dataset is empty.")

# ================================================================
# Verification Functions for Image Integrity
# ================================================================

def verify_csv_images(csv_path, images_dir, split_name=""):
    """
    Verify that all images listed in the CSV exist in the specified directory.
    
    Args:
        csv_path (str): Path to the CSV file.
        images_dir (str): Directory where images are stored.
        split_name (str): Name of the dataset split (for logging purposes).
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Cannot verify images.")
        return
    
    df = pd.read_csv(csv_path)
    missing_images = []
    for img in df['image']:
        img_path = os.path.join(images_dir, os.path.basename(img))
        if not os.path.exists(img_path):
            missing_images.append(img)
    
    if missing_images:
        print(f"{split_name} Split: {len(missing_images)} images are missing.")
        # Log missing images
        with open('missing_images.log', 'a') as log_file:
            for img in missing_images:
                log_file.write(f"Missing image: {img}\n")
    else:
        print(f"{split_name} Split: All images are present.")

def check_corrupted_images(csv_path, images_dir, split_name=""):
    """
    Check for corrupted images in the dataset.
    
    Args:
        csv_path (str): Path to the CSV file.
        images_dir (str): Directory where images are stored.
        split_name (str): Name of the dataset split (for logging purposes).
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Cannot check images.")
        return
    
    df = pd.read_csv(csv_path)
    corrupted_images = []
    for img in df['image']:
        img_path = os.path.join(images_dir, os.path.basename(img))
        try:
            with Image.open(img_path) as image:
                image.verify()  # Verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            corrupted_images.append(img)
    
    if corrupted_images:
        print(f"{split_name} Split: {len(corrupted_images)} images are corrupted.")
        # Log corrupted images
        with open('corrupted_images.log', 'a') as log_file:
            for img in corrupted_images:
                log_file.write(f"Corrupted image: {img}\n")
    else:
        print(f"{split_name} Split: No corrupted images found.")

# Verify images in the training split
verify_csv_images(train_split_csv, train_dir, split_name="Training")

# Verify images in the validation split
verify_csv_images(valid_split_csv, valid_dir, split_name="Validation")

# Verify images in the test split if available
if test_split_csv and os.path.exists(test_split_csv):
    verify_csv_images(test_split_csv, valid_dir, split_name="Test")

# Check for corrupted images in the training split
check_corrupted_images(train_split_csv, train_dir, split_name="Training")

# Check for corrupted images in the validation split
check_corrupted_images(valid_split_csv, valid_dir, split_name="Validation")

# Check for corrupted images in the test split if available
if test_split_csv and os.path.exists(test_split_csv):
    check_corrupted_images(test_split_csv, valid_dir, split_name="Test")