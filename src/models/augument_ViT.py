# src/models/ViT.py

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# tqdm for progress bars
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds, accuracy_fn  # Custom helper functions
from helper_functions import *

# ================================================================
# Setup Logging
# ================================================================
logging.basicConfig(
    filename='missing_images.log',
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
BATCH_SIZE = 64        # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 20        # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement

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

# Define directories for data, results, figures, and models
data_path = os.path.join(
    project_root,
    "data",
    "processed",
    "plant_leaf_disease_dataset",
    "single_task_disease",
)
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Define output directories
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
# Data Transforms with Class-Specific Augmentations
# ================================================================

# Define transforms for majority classes
transform_major = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # Normalize if needed
    # transforms.Normalize(
    #     [0.485, 0.456, 0.406],  # Mean for ImageNet
    #     [0.229, 0.224, 0.225]   # Std for ImageNet
    # ),
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
    # Normalize if needed
    # transforms.Normalize(
    #     [0.485, 0.456, 0.406],
    #     [0.229, 0.224, 0.225]
    # ),
])

# ================================================================
# Initialize Datasets and DataLoaders
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

# Create WeightedRandomSampler to balance class representation
# Compute class counts
class_counts = train_df['label'].value_counts().sort_index().values
num_classes = len(class_counts)
class_weights = 1. / class_counts
sample_weights = train_df['label'].map(lambda x: class_weights[x]).values

# Create sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,  # Use sampler instead of shuffle
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
# Data Visualization: Plotting Label Distribution
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
    plt.figure(figsize=(14, 8))
    label_counts.plot(kind='bar', color='skyblue')
    plt.title(f"Label Distribution in {dataset_name} Dataset")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Save the plot in the figures directory
    plt.savefig(os.path.join(output_dirs[1], f"label_distribution_{dataset_name.lower()}.pdf"))
    plt.show()

# # Plot label distribution for training and validation sets
# plot_label_distribution_pandas(train_split_csv, idx_to_disease, "Training")
# plot_label_distribution_pandas(valid_split_csv, idx_to_disease, "Validation")
# if test_split_csv and os.path.exists(test_split_csv):
#     plot_label_distribution_pandas(test_split_csv, idx_to_disease, "Test")

# ================================================================
# Visualization Functions to Verify Augmentation
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
                # # Unnormalize the image for display
                # mean = np.array([0.485, 0.456, 0.406])
                # std = np.array([0.229, 0.224, 0.225])
                # image = std * image + mean
                # image = np.clip(image, 0, 1)
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
                # transforms.Normalize(
                #     [0.485, 0.456, 0.406],   # Mean for ImageNet
                #     [0.229, 0.224, 0.225]    # Std for ImageNet
                # ),
            ])
            img_name = dataset.annotations.iloc[idx]['image']
            img_path = os.path.join(dataset.images_dir, os.path.basename(img_name))
            try:
                original_image = Image.open(img_path).convert("RGB")
                original_image = original_transform(original_image).numpy().transpose((1, 2, 0))
                # Unnormalize
                # mean = np.array([0.485, 0.456, 0.406])
                # std = np.array([0.229, 0.224, 0.225])
                # original_image = std * original_image + mean
                # original_image = np.clip(original_image, 0, 1)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # Augmented Image
            augmented_image = img.numpy().transpose((1, 2, 0))
            # # Unnormalize
            # augmented_image = std * augmented_image + mean
            # augmented_image = np.clip(augmented_image, 0, 1)
            
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
    # Unnormalize
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = std * image + mean
    # image = np.clip(image, 0, 1)

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    label_name = idx_to_disease.get(label.item(), "Unknown")
    plt.title(f"Label: {label_name}")
    plt.axis('off')
    # Save the plot in the figures directory
    plt.savefig(os.path.join(output_dirs[1], "random_image_from_loader.pdf"))
    plt.show()

# # Plot a random image from the train_loader
# if len(train_dataset) > 0:
#     plot_random_image_from_loader(train_loader, idx_to_disease, output_dirs)
# else:
#     print("Cannot plot image: Training dataset is empty.")

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

# ================================================================
# Comprehensive Visualization Function
# ================================================================

def comprehensive_verification(dataset, idx_to_disease, minority_classes, num_samples=5):
    """
    Perform comprehensive verification of data augmentations.
    
    Args:
        dataset (Dataset): The dataset to inspect.
        idx_to_disease (dict): Mapping from label index to disease name.
        minority_classes (list): List of minority class indices.
        num_samples (int): Number of samples to plot per class.
    
    Returns:
        None
    """
    for cls in minority_classes:
        # Collect samples
        samples = []
        for img, label in dataset:
            if label.item() == cls:
                samples.append(img)
                if len(samples) >= num_samples:
                    break
        if not samples:
            print(f"No samples found for class {cls} ({idx_to_disease.get(cls, 'Unknown')}).")
            continue
        
        # Plot samples
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Augmented Samples for Class: {idx_to_disease.get(cls, 'Unknown')}", fontsize=16)
        for i, img in enumerate(samples):
            image = img.numpy().transpose((1, 2, 0))
            # # Unnormalize
            # mean = np.array([0.485, 0.456, 0.406])
            # std = np.array([0.229, 0.224, 0.225])
            # image = std * image + mean
            # image = np.clip(image, 0, 1)
            plt.subplot(1, num_samples, i+1)
            plt.imshow(image)
            plt.axis('off')
        # Save the plot in the figures directory
        plt.savefig(os.path.join(output_dirs[1], f"comprehensive_verification_class_{cls}.pdf"))
        plt.show()
        
        # Compute statistics
        image_array = np.array([img.numpy() for img in samples])
        mean = image_array.mean(axis=(0, 2, 3))
        std = image_array.std(axis=(0, 2, 3))
        print(f"Class {cls} ({idx_to_disease.get(cls, 'Unknown')}): Mean={mean}, Std={std}\n")

# Usage
# comprehensive_verification(train_dataset, idx_to_disease, minority_classes, num_samples=5)

# ================================================================
# Vision Transformer (ViT) Architecture
# ================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Using a Conv2d layer to perform patch extraction and embedding
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size]
        
        Returns:
            Tensor: Patch embeddings of shape [batch_size, num_patches, embed_dim]
        """
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches**0.5, num_patches**0.5]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension of the embedding space.
            num_patches (int): Number of patches in the input.
            dropout (float): Dropout rate.
        """
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize the [CLS] token and positional embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, num_patches, embed_dim]
        
        Returns:
            Tensor: Positionally encoded tensor of shape [batch_size, num_patches + 1, embed_dim]
        """
        batch_size, num_patches, embed_dim = x.size()
        
        # [CLS] token: a learnable embedding prepended to the patch embeddings
        cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)).to(x.device)
        cls_token = cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        
        # Concatenate [CLS] token with patch embeddings
        x = torch.cat((cls_token, x), dim=1)  # Shape: [batch_size, num_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, num_tokens, embed_dim]
        
        Returns:
            Tensor: Output tensor of shape [batch_size, num_tokens, embed_dim]
        """
        batch_size, num_tokens, embed_dim = x.size()
        
        # Linear projection and split into Q, K, V
        qkv = self.qkv(x)  # Shape: [batch_size, num_tokens, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, batch_size, num_heads, num_tokens, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: [batch_size, num_heads, num_tokens, head_dim]
        
        # Compute scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [batch_size, num_heads, num_tokens, num_tokens]
        attn_probs = attn_scores.softmax(dim=-1)  # Shape: [batch_size, num_heads, num_tokens, num_tokens]
        attn_probs = self.attn_dropout(attn_probs)
        
        # Weighted sum of values
        attn_output = attn_probs @ v  # Shape: [batch_size, num_heads, num_tokens, head_dim]
        attn_output = attn_output.transpose(1, 2)  # Shape: [batch_size, num_tokens, num_heads, head_dim]
        attn_output = attn_output.flatten(2)  # Shape: [batch_size, num_tokens, embed_dim]
        
        # Final linear projection
        out = self.proj(attn_output)  # Shape: [batch_size, num_tokens, embed_dim]
        out = self.proj_dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension of the embedding space.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, num_tokens, embed_dim]
        
        Returns:
            Tensor: Output tensor of shape [batch_size, num_tokens, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of the hidden dimension in FFN to embed_dim.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, num_tokens, embed_dim]
        
        Returns:
            Tensor: Output tensor of shape [batch_size, num_tokens, embed_dim]
        """
        # MHSA block with residual connection
        x = x + self.mhsa(self.norm1(x))
        
        # FFN block with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.1,
    ):
        """
        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_classes (int): Number of output classes.
            embed_dim (int): Dimension of the embedding space.
            depth (int): Number of transformer encoder blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of the hidden dimension in FFN to embed_dim.
            dropout (float): Dropout rate.
        """
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = PositionalEncoding(embed_dim, num_patches, dropout)
        
        # Transformer Encoder Blocks
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification Head
        self.cls_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size]
        
        Returns:
            Tensor: Logits of shape [batch_size, num_classes]
        """
        x = self.patch_embed(x)  # Shape: [batch_size, num_patches, embed_dim]
        x = self.pos_embed(x)    # Shape: [batch_size, num_patches + 1, embed_dim]
        x = self.transformer(x)  # Shape: [batch_size, num_patches + 1, embed_dim]
        x = self.norm(x)         # Shape: [batch_size, num_patches + 1, embed_dim]
        
        # [CLS] token is the first token
        cls_token = x[:, 0]      # Shape: [batch_size, embed_dim]
        logits = self.cls_head(cls_token)  # Shape: [batch_size, num_classes]
        return logits

# ================================================================
# Model Initialization
# ================================================================

# Initialize Vision Transformer model from scratch
model = VisionTransformer(
    img_size=HEIGHT,
    patch_size=16,
    in_channels=3,
    num_classes=len(disease_to_idx),
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
)

# Move the model to the configured device
model = model.to(device)

# If multiple GPUs are available, use DataParallel
if num_gpus > 1:
    model = nn.DataParallel(model)

# ================================================================
# Loss Function and Optimizer
# ================================================================

# Define loss function with class weights to handle imbalance
# Compute class weights inversely proportional to class frequencies
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1. / class_counts
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define loss function
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Define optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ================================================================
# Training and Validation Functions
# ================================================================

from sklearn.metrics import classification_report, confusion_matrix

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to train on.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device, collect_metrics=False):
    """
    Validates the model.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to validate on.
        collect_metrics (bool): If True, collect labels and predictions.

    Returns:
        tuple: (epoch_loss, epoch_accuracy, all_labels, all_preds) if collect_metrics=True
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            if collect_metrics:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return (epoch_loss, epoch_acc.item(), all_labels, all_preds) if collect_metrics else (epoch_loss, epoch_acc.item())

# ================================================================
# Training Loop
# ================================================================

import time

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Early Stopping variables
best_val_acc = 0.0
trigger_times = 0

# Time tracking
total_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 10)
    
    epoch_start_time = time.time()
    
    # Training Phase
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    
    # Validation Phase
    val_loss, val_acc = validate(model, valid_loader, criterion, device)
    print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}%")
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch Duration: {epoch_duration:.2f} seconds")
    
    # Append metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Learning rate scheduler step
    scheduler.step()
    
    # Check for improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(project_root, "models", "best_vit_single_task_disease.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at {best_model_path}!")
        trigger_times = 0
    else:
        trigger_times += 1
        print(f"No improvement in validation accuracy. Trigger times: {trigger_times}")
        if trigger_times >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered!")
            break

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"\nTotal Training Time: {total_duration/60:.2f} minutes")

# ================================================================
# Visualization Utilities
# ================================================================

def imshow_batch(images, labels, idx_to_disease, classes_per_row=4, output_dirs=None):
    """
    Display a batch of images with their corresponding labels.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): Corresponding labels.
        idx_to_disease (dict): Mapping from index to disease name.
        classes_per_row (int): Number of images per row in the grid.
        output_dirs (list): List of output directories for saving plots.
    """
    # Unnormalize the images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = images.cpu().numpy().transpose((0, 2, 3, 1))
    images = std * images + mean
    images = np.clip(images, 0, 1)
    
    # Determine grid size
    batch_size = images.shape[0]
    num_rows = batch_size // classes_per_row + int(batch_size % classes_per_row != 0)
    
    plt.figure(figsize=(classes_per_row * 3, num_rows * 3))
    for idx in range(batch_size):
        plt.subplot(num_rows, classes_per_row, idx + 1)
        plt.imshow(images[idx])
        label = idx_to_disease.get(labels[idx].item(), "Unknown")
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    if output_dirs:
        plt.savefig(os.path.join(output_dirs[1], "training_batch_images.pdf"))
    plt.show()


# ================================================================
# Visualization of Training Data
# ================================================================

# Get a batch of training data
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Display the images with labels
imshow_batch(images, labels, idx_to_disease, classes_per_row=4, output_dirs=output_dirs)

def imshow_transforms(dataset, idx, idx_to_disease, num_transforms=5, output_dirs=None):
    """
    Display original and augmented versions of a single image.

    Args:
        dataset (Dataset): The dataset from which to retrieve the image.
        idx (int): Index of the image in the dataset.
        idx_to_disease (dict): Mapping from index to disease name.
        num_transforms (int): Number of augmented versions to display.
        output_dirs (list): List of output directories for saving plots.
    """
    # Retrieve image and label
    image, label = dataset[idx]
    img_name = dataset.annotations.iloc[idx]['image']
    img_path = os.path.join(dataset.images_dir, os.path.basename(img_name))
    
    # Open original image
    original_image = Image.open(img_path).convert('RGB').resize((HEIGHT, WIDTH))
    
    # Define a temporary transform with augmentation
    temp_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],   # Mean for ImageNet
            [0.229, 0.224, 0.225]    # Std for ImageNet
        ),
    ])
    
    # Apply the temporary transform multiple times to get different augmentations
    augmented_images = [temp_transform(original_image) for _ in range(num_transforms)]
    
    # Unnormalize the original image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image_np = np.array(original_image) / 255.0  # Normalize to [0,1]
    
    # Unnormalize augmented images
    augmented_images = [img.cpu().numpy().transpose((1, 2, 0)) for img in augmented_images]
    augmented_images = [std * img + mean for img in augmented_images]
    augmented_images = [np.clip(img, 0, 1) for img in augmented_images]
    
    # Plot original and augmented images
    plt.figure(figsize=(15, 3))
    
    # Original Image
    plt.subplot(1, num_transforms + 1, 1)
    plt.imshow(original_image_np)
    label_name = idx_to_disease.get(label.item(), "Unknown")
    plt.title(f"Original: {label_name}")
    plt.axis('off')
    
    # Augmented Images
    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, num_transforms + 1, i + 2)
        plt.imshow(aug_img)
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    if output_dirs:
        plt.savefig(os.path.join(output_dirs[1], f"transforms_image_{idx}.pdf"))
    plt.show()

# ================================================================
# Visualize Data Augmentation on a Single Image
# ================================================================

# Choose an index to visualize (e.g., first image in the training dataset)
image_idx = 0

imshow_transforms(train_dataset, image_idx, idx_to_disease, num_transforms=5, output_dirs=output_dirs)

def visualize_single_image_flow(dataset, idx, model, device, idx_to_disease, output_dirs=None):
    """
    Visualize the flow of a single image from dataset to model prediction.

    Args:
        dataset (Dataset): The dataset from which to retrieve the image.
        idx (int): Index of the image in the dataset.
        model (nn.Module): The trained model.
        device (torch.device): Device to perform computations on.
        idx_to_disease (dict): Mapping from index to disease name.
        output_dirs (list): List of output directories for saving plots.
    """
    # Retrieve image and label from dataset
    image, label = dataset[idx]
    img_name = dataset.annotations.iloc[idx]['image']
    img_path = os.path.join(dataset.images_dir, os.path.basename(img_name))
    
    # Display the image
    plt.figure(figsize=(3,3))
    original_image = Image.open(img_path).convert('RGB').resize((HEIGHT, WIDTH))
    image_np = np.array(original_image) / 255.0  # Normalize to [0,1]
    plt.imshow(image_np)
    label_name = idx_to_disease.get(label.item(), "Unknown")
    plt.title(f"True Label: {label_name}")
    plt.axis('off')
    if output_dirs:
        plt.savefig(os.path.join(output_dirs[1], f"single_image_flow_{idx}.pdf"))
    plt.show()
    
    # Add batch dimension and move to device
    input_tensor = image.unsqueeze(0).to(device)  # Shape: [1, C, H, W]
    
    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # Shape: [1, num_classes]
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Get predicted label
    predicted_label = idx_to_disease.get(predicted_idx.item(), "Unknown")
    confidence = confidence.item() * 100
    
    # Display prediction
    print(f"Predicted Label: {predicted_label} ({confidence:.2f}% confidence)")
    
    # Plot the probabilities as a bar chart
    plt.figure(figsize=(10,4))
    classes = list(disease_to_idx.keys())
    probs = probabilities.cpu().numpy().flatten()
    plt.barh(classes, probs, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.gca().invert_yaxis()  # Highest probability on top
    plt.tight_layout()
    if output_dirs:
        plt.savefig(os.path.join(output_dirs[1], f"prediction_probabilities_{idx}.pdf"))
    plt.show()

# ================================================================
# Visualize Flow of a Single Image Through the Model
# ================================================================

# Choose an index to visualize (e.g., first image in the training dataset)
single_image_idx = 0

visualize_single_image_flow(train_dataset, single_image_idx, model, device, idx_to_disease, output_dirs)

# ================================================================
# Plot Training and Validation Metrics
# ================================================================

def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        train_losses (list): List of training losses.
        train_accuracies (list): List of training accuracies.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training metrics plot saved at {save_path}")
    else:
        plt.show()

# ================================================================
# Plotting Training and Validation Metrics
# ================================================================

# After training loop
plot_training_metrics(
    train_losses, 
    train_accuracies, 
    val_losses, 
    val_accuracies, 
    save_path=os.path.join(output_dirs[1], "training_validation_metrics.png")
)

# ================================================================
# Post-Training Evaluation: Classification Report and Confusion Matrix
# ================================================================

def evaluate_model_post_training(model, dataloader, device, idx_to_disease, output_dirs=None):
    """
    Evaluate the model on a dataset and print classification metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on.
        idx_to_disease (dict): Mapping from index to disease name.
        output_dirs (list): List of output directories for saving plots.

    Returns:
        None
    """
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    total_samples = 0

    # Assuming the same loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Post-Training Evaluation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    print(f"\nPost-Training Evaluation Loss: {avg_loss:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(disease_to_idx.keys())))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(disease_to_idx.keys()), yticklabels=list(disease_to_idx.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if output_dirs:
        plt.savefig(os.path.join(output_dirs[1], "post_training_confusion_matrix.pdf"))
    plt.show()

# Perform post-training evaluation on the validation set
evaluate_model_post_training(model, valid_loader, device, idx_to_disease, output_dirs)

# Optionally, evaluate on the test set if available
if test_loader:
    print("\nEvaluating on Test Set:")
    evaluate_model_post_training(model, test_loader, device, idx_to_disease, output_dirs)
else:
    print("\nNo test set available for evaluation.")

# ================================================================
# Final Recommendations and Best Practices
# ================================================================

# - **Model Saving:** The best model (based on validation accuracy) is already being saved during training.
# - **Subset Selection for Testing:** You can utilize the test set initialized above for final evaluation.
# - **Early Stopping:** Implemented with patience; adjust `EARLY_STOPPING_PATIENCE` as needed.
# - **Monitoring Training:** Utilize the plots generated to monitor overfitting or underfitting.
# - **Further Enhancements:**
#     - Implement callbacks for more sophisticated training monitoring.
#     - Explore more advanced learning rate schedulers.
#     - Experiment with different augmentation strategies.
#     - Incorporate model checkpointing based on validation loss as well.

# ================================================================
# End of Script
# ================================================================
