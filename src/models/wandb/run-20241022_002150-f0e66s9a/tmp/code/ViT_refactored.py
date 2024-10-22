# train_vit_model.py

# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import wandb
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# Helper Functions and Settings
# ================================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import set_seeds
from helper_functions import *

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

# W&B Project Name
WANDB_PROJECT_NAME = "Plant_Leaf_Disease_ViT"
model_name = "ViT"  # Define model name globally

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

# Define project root (assuming this script is in the project root)
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

# Define output directories for results, figures, models, and logs
output_dirs = {
    "results": os.path.join(project_root, "reports", "results"),
    "figures": os.path.join(project_root, "reports", "figures"),
    "models": os.path.join(project_root, "models", "ViT"),
    "logs": os.path.join(project_root, "logs"),
}

# Create output directories if they don't exist
for directory in output_dirs.values():
    os.makedirs(directory, exist_ok=True)

# Setup Logging
log_file = os.path.join(output_dirs["logs"], 'vit_training.log')
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to list directory contents
def list_directory_contents(directory, num_items=10):
    if os.path.exists(directory):
        contents = os.listdir(directory)
        print(f"Contents of {directory} ({len(contents)} items): {contents[:num_items]}...")
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

# Path to training split CSV
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
# Custom Dataset Class
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
# Data Transforms with Class-Specific Augmentations
# ================================================================

# Define transforms for majority classes
transform_major = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
    #     std=[0.229, 0.224, 0.225]     # Std for ImageNet
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
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],   # Mean for ImageNet
    #     std=[0.229, 0.224, 0.225]     # Std for ImageNet
    # ),
])

# ================================================================
# Initialize Datasets and DataLoaders (Using WeightedRandomSampler)
# ================================================================

# Path to validation split CSV
valid_split_csv = os.path.join(data_path, "valid_split.csv")
if os.path.exists(valid_split_csv):
    valid_df = pd.read_csv(valid_split_csv)
    valid_dataset = PlantDiseaseDataset(
        csv_file=valid_split_csv,
        images_dir=valid_dir,
        transform_major=transform_major,  # Validation should not have augmentation
        transform_minority=None,          # No augmentation for validation
        minority_classes=[],              # No augmentation needed
        image_col='image',
        label_col='label'
    )
else:
    print(f"Error: Validation split CSV not found at {valid_split_csv}. Exiting.")
    sys.exit(1)

# Initialize training dataset
train_dataset = PlantDiseaseDataset(
    csv_file=train_split_csv,
    images_dir=train_dir,
    transform_major=transform_major,
    transform_minority=transform_minority,
    minority_classes=minority_classes,
    image_col='image',
    label_col='label'
)

# Create WeightedRandomSampler for the training DataLoader
# Compute class counts and weights
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1. / class_counts
samples_weight = class_weights[train_df['label'].values]
samples_weight = torch.from_numpy(samples_weight).double()

# Create the sampler
sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

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

# Display dataset information
print(f"\nNumber of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
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
        
        # Initialize the positional embeddings
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

# Define the number of classes
output_size = len(disease_to_idx)

# Initialize Vision Transformer model from scratch
vit_model = VisionTransformer(
    img_size=HEIGHT,
    patch_size=16,
    in_channels=3,
    num_classes=output_size,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
)

# Move the model to the configured device
vit_model = vit_model.to(device)

# If multiple GPUs are available, use DataParallel
if num_gpus > 1:
    vit_model = nn.DataParallel(vit_model)

# ================================================================
# Loss Function and Optimizer
# ================================================================

# Remove class weighting from loss function since we're using WeightedRandomSampler
criterion = nn.CrossEntropyLoss()

# Define optimizer with a lower learning rate for training from scratch
optimizer = optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ================================================================
# Callbacks for Training Monitoring
# ================================================================

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0.0, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

class ModelCheckpoint:
    """
    Saves the model based on validation loss.
    """
    def __init__(self, path='best_val_loss_model.pth', verbose=False):
        """
        Args:
            path (str): Path to save the model.
            verbose (bool): If True, prints messages when saving the model.
        """
        self.best_loss = None
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            if self.verbose:
                print(f"Validation loss improved ({self.best_loss if self.best_loss else 'N/A'} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)

# ================================================================
# Training and Validation Functions
# ================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, log_interval=10):
    """
    Trains the model for one epoch using mixed precision.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to train on.
        scaler (GradScaler): GradScaler for mixed precision.
        epoch (int): Current epoch number.
        log_interval (int): How often to log batch metrics.
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        # Enhanced Logging: Log every 'log_interval' batches
        if (batch_idx + 1) % log_interval == 0:
            unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
            class_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}
            wandb.log({
                f"{model_name}/train_loss": loss.item(),
                f"{model_name}/batch_train_accuracy": torch.sum(preds == labels.data).item() / inputs.size(0),
                f"{model_name}/batch_class_distribution": class_distribution
            })
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f} | Class Distribution: {class_distribution}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    wandb.log({
        f"{model_name}/epoch_train_loss": epoch_loss,
        f"{model_name}/epoch_train_accuracy": epoch_acc.item()
    })

    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device, collect_metrics=True):
    """
    Validates the model.
    
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to validate on.
        collect_metrics (bool): If True, collect labels and predictions.
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy, all_labels, all_preds)
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

            # Mixed precision inference
            with autocast():
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

    wandb.log({
        f"{model_name}/epoch_val_loss": epoch_loss,
        f"{model_name}/epoch_val_accuracy": epoch_acc.item()
    })

    return epoch_loss, epoch_acc.item(), all_labels, all_preds

# ================================================================
# Visualization Utilities
# ================================================================

def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies, model_name, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        train_losses (list): List of training losses.
        train_accuracies (list): List of training accuracies.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        wandb.log({f"{model_name}/training_validation_metrics": wandb.Image(save_path)})
        print(f"Training metrics plot saved at {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_model_post_training(model, dataloader, device, idx_to_disease, model_name, save_dir):
    """
    Evaluate the model on a dataset and print classification metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on.
        idx_to_disease (dict): Mapping from index to disease name.
        model_name (str): Name of the model for reporting.
        save_dir (str): Directory to save the confusion matrix plot.

    Returns:
        None
    """
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    total_samples = 0

    # Define loss function (same as training)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{model_name} - Evaluation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Mixed precision inference
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    print(f"\n{model_name} - Evaluation Loss: {avg_loss:.4f}")

    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=list(idx_to_disease.values()), output_dict=True)
    report = classification_report(all_labels, all_preds, target_names=list(idx_to_disease.values()))
    print(f"\n{model_name} - Classification Report:")
    print(report)

    # Save classification report as JSON
    report_save_path = os.path.join(output_dirs["results"], f"{model_name}_classification_report.json")
    with open(report_save_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved at {report_save_path}")

    wandb.log({f"{model_name}/classification_report": wandb.Table(dataframe=pd.DataFrame(report_dict).transpose())})

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(idx_to_disease.values()), 
                yticklabels=list(idx_to_disease.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    # Save the plot
    cm_save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_save_path)
    wandb.log({f"{model_name}/confusion_matrix": wandb.Image(cm_save_path)})
    plt.close()
    print(f"Confusion matrix saved at {cm_save_path}")

    # Save confusion matrix data
    cm_data_save_path = os.path.join(output_dirs["results"], f"{model_name}_confusion_matrix.npy")
    np.save(cm_data_save_path, cm)
    print(f"Confusion matrix data saved at {cm_data_save_path}")

# ================================================================
# Initialize W&B
# ================================================================

# Initialize W&B run
wandb.init(
    project=WANDB_PROJECT_NAME,
    name="ViT_Training",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "model": "VisionTransformer",
        "optimizer": "AdamW",
        "scheduler": "StepLR",
        "num_classes": output_size,
        "image_size": f"{HEIGHT}x{WIDTH}",
    },
    save_code=True
)

# Get the run id for tracking
run_id = wandb.run.id
print(f"W&B Run ID: {run_id}")

# ================================================================
# Callbacks for Training Monitoring
# ================================================================

# Initialize EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, 
    verbose=True, 
    path=os.path.join(output_dirs[2], "best_val_loss_model.pth")
)
model_checkpoint = ModelCheckpoint(
    path=os.path.join(output_dirs[2], "best_val_loss_model.pth"), 
    verbose=True
)

# ================================================================
# Initialize GradScaler for Mixed Precision
# ================================================================
scaler = GradScaler()

# ================================================================
# Training Loop with Callbacks and Mixed Precision
# ================================================================

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Time tracking
total_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 30)
    
    epoch_start_time = time.time()
    
    # Training Phase
    train_loss, train_acc = train_one_epoch(
        model=vit_model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        epoch=epoch
    )
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    
    # Validation Phase
    val_loss, val_acc, _, _ = validate(
        model=vit_model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device
    )
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
    
    # Log learning rate
    current_lr = scheduler.get_last_lr()[0]
    wandb.log({f"{vit_model.__class__.__name__}/learning_rate": current_lr})
    
    # Model checkpoint based on validation loss
    model_checkpoint(val_loss, vit_model)
    
    # Early Stopping based on validation loss
    early_stopping(val_loss, vit_model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"\nTotal Training Time: {total_duration/60:.2f} minutes")

# Log total training time to W&B
wandb.log({"total_training_time_minutes": total_duration/60})

# ================================================================
# Save Metrics to Results Folder
# ================================================================

# Save metrics to CSV files
metrics_df = pd.DataFrame({
    'epoch': range(1, len(train_losses)+1),
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})
metrics_save_path = os.path.join(output_dirs["results"], f'{model_name}_training_metrics.csv')
metrics_df.to_csv(metrics_save_path, index=False)
print(f"Training metrics saved at {metrics_save_path}")

# ================================================================
# Visualization and Saving Artifacts
# ================================================================

# Plot training metrics
plot_save_path = os.path.join(output_dirs["figures"], f"{model_name}_training_validation_metrics.png")
plot_training_metrics(
    train_losses, 
    train_accuracies, 
    val_losses, 
    val_accuracies, 
    model_name=model_name,
    save_path=plot_save_path
)

# Perform post-training evaluation on the validation set
evaluate_model_post_training(
    model=vit_model, 
    dataloader=valid_loader, 
    device=device, 
    idx_to_disease=idx_to_disease, 
    model_name=model_name, 
    save_dir=output_dirs["figures"]
)

# Save W&B artifacts
wandb.finish()