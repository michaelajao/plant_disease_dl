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
import plotly.express as px

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

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
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Hyperparameters
BATCH_SIZE = 32        # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 20        # Number of epochs for training
HEIGHT, WIDTH = 224, 224  # Image dimensions

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
# Custom Dataset Class
# ================================================================

class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, image_col='image', label_col='label'):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_col (str): Column name for image filenames in the CSV.
            label_col (str): Column name for labels in the CSV.
        """
        self.annotations = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

        # Verify required columns
        required_columns = [image_col, label_col]
        for col in required_columns:
            if col not in self.annotations.columns:
                raise ValueError(f"Missing required column '{col}' in CSV file.")

        self.image_col = image_col
        self.label_col = label_col

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

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# ================================================================
# Split Dataset into Training and Validation Sets
# ================================================================

# Paths to CSV files
full_csv = os.path.join(data_path, "dataset_single_task_disease.csv")
train_split_csv = os.path.join(data_path, "train_split.csv")
valid_split_csv = os.path.join(data_path, "valid_split.csv")

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
    print("Dataset split based on 'split' column.")
else:
    # If no 'split' column, perform an 80-20 split
    from sklearn.model_selection import train_test_split

    if 'label' not in full_df.columns:
        raise ValueError("CSV file must contain a 'label' column for stratified splitting.")

    train_df, valid_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["label"]
    )
    print("Dataset split into 80% training and 20% validation.")

# Save the split CSVs
train_df.to_csv(train_split_csv, index=False)
valid_df.to_csv(valid_split_csv, index=False)
print(f"Saved training split to {train_split_csv} with {len(train_df)} samples.")
print(f"Saved validation split to {valid_split_csv} with {len(valid_df)} samples.")

# ================================================================
# Data Transforms
# ================================================================

# Define transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # Uncomment if you decide to apply normalization
    # transforms.Normalize(
    #     [0.485, 0.456, 0.406],  # Mean for ImageNet
    #     [0.229, 0.224, 0.225]   # Std for ImageNet
    # ),
])

valid_transforms = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    # Uncomment if you decide to apply normalization
    # transforms.Normalize(
    #     [0.485, 0.456, 0.406],  # Mean for ImageNet
    #     [0.229, 0.224, 0.225]   # Std for ImageNet
    # ),
])

# ================================================================
# Initialize Datasets and DataLoaders
# ================================================================

# Initialize training and validation datasets
train_dataset = PlantDiseaseDataset(
    csv_file=train_split_csv,
    images_dir=train_dir,
    transform=train_transforms,
    image_col='image',  # Ensure this matches the actual column name in your CSV
    label_col='label'   # Ensure this matches the actual column name in your CSV
)

valid_dataset = PlantDiseaseDataset(
    csv_file=valid_split_csv,
    images_dir=valid_dir,
    transform=valid_transforms,
    image_col='image',  # Ensure this matches the actual column name in your CSV
    label_col='label'   # Ensure this matches the actual column name in your CSV
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
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
# Data Visualization for a Single Batch with Random Image Selection
# ================================================================

def plot_random_image_from_loader(dataloader, idx_to_disease):
    """
    Plot a random image from the dataloader with its label.

    Args:
        dataloader (DataLoader): DataLoader to fetch the image from.
        idx_to_disease (dict): Mapping from index to disease name.
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

    # Since normalization is not applied, no need to unnormalize
    # If you decide to apply normalization, uncomment and adjust the following lines
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
    plt.show()

# Plot a random image from the train_loader
if len(train_dataset) > 0:
    plot_random_image_from_loader(train_loader, idx_to_disease)
else:
    print("Cannot plot image: Training dataset is empty.")

# ================================================================
# Function to Check Label Distribution and Plot It
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
    label_counts.plot(kind='bar', figsize=(14, 8), color='skyblue')
    plt.title(f"Label Distribution in {dataset_name} Dataset")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Plot label distribution for training and validation sets
plot_label_distribution_pandas(train_split_csv, idx_to_disease, "Training")

plot_label_distribution_pandas(valid_split_csv, idx_to_disease, "Validation")

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

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Optional: Define a learning rate scheduler
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

def validate(model, dataloader, criterion, device):
    """
    Validates the model.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to validate on.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
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

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(disease_to_idx.keys())))

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(disease_to_idx.keys()), yticklabels=list(disease_to_idx.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return epoch_loss, epoch_acc.item()


# ================================================================
# Training Loop
# ================================================================

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
patience = 5
trigger_times = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 10)
    
    # Training Phase
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    
    # Validation Phase
    val_loss, val_acc = validate(model, valid_loader, criterion, device)
    print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}%")
    
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
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")


# ================================================================
# Visualization Utilities
# ================================================================

def imshow_batch(images, labels, idx_to_disease, classes_per_row=4):
    """
    Display a batch of images with their corresponding labels.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): Corresponding labels.
        idx_to_disease (dict): Mapping from index to disease name.
        classes_per_row (int): Number of images per row in the grid.
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
    plt.show()


# ================================================================
# Visualization of Training Data
# ================================================================

# Get a batch of training data
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Display the images with labels
imshow_batch(images, labels, idx_to_disease, classes_per_row=4)


def imshow_transforms(dataset, idx, idx_to_disease, num_transforms=5):
    """
    Display original and augmented versions of a single image.

    Args:
        dataset (Dataset): The dataset from which to retrieve the image.
        idx (int): Index of the image in the dataset.
        idx_to_disease (dict): Mapping from index to disease name.
        num_transforms (int): Number of augmented versions to display.
    """
    # Retrieve image and label
    image, label = dataset[idx]
    img_name = dataset.annotations.iloc[idx]['filename']
    img_path = os.path.join(dataset.images_dir, img_name)
    
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
    plt.show()


# ================================================================
# Visualize Data Augmentation on a Single Image
# ================================================================

# Choose an index to visualize (e.g., first image in the training dataset)
image_idx = 0

imshow_transforms(train_dataset, image_idx, idx_to_disease, num_transforms=5)


def visualize_single_image_flow(dataset, idx, model, device, idx_to_disease):
    """
    Visualize the flow of a single image from dataset to model prediction.

    Args:
        dataset (Dataset): The dataset from which to retrieve the image.
        idx (int): Index of the image in the dataset.
        model (nn.Module): The trained model.
        device (torch.device): Device to perform computations on.
        idx_to_disease (dict): Mapping from index to disease name.
    """
    # Retrieve image and label from dataset
    image, label = dataset[idx]
    img_name = dataset.annotations.iloc[idx]['filename']
    img_path = os.path.join(dataset.images_dir, img_name)
    
    # Display the image
    plt.figure(figsize=(3,3))
    original_image = Image.open(img_path).convert('RGB').resize((HEIGHT, WIDTH))
    image_np = np.array(original_image) / 255.0  # Normalize to [0,1]
    plt.imshow(image_np)
    label_name = idx_to_disease.get(label.item(), "Unknown")
    plt.title(f"True Label: {label_name}")
    plt.axis('off')
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
    plt.show()


# ================================================================
# Visualize Flow of a Single Image Through the Model
# ================================================================

# Choose an index to visualize (e.g., first image in the training dataset)
single_image_idx = 0

visualize_single_image_flow(train_dataset, single_image_idx, model, device, idx_to_disease)


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
    save_path=os.path.join(project_root, "reports", "figures", "training_validation_metrics.png")
)
