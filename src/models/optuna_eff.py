# %%
import os
import pandas as pd
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
# Load the train , test and validation data and labels
print(os.listdir("../../data/raw/Food"))
labels_df = pd.read_csv("../../data/raw/Food/labels/labels.csv")
# Define the data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, dataframe, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Load image files
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        )
        # Initialize a dictionary to map frame identifiers to labels
        self.labels_map = {}
        # Populate the labels_map
        for _, row in dataframe.iterrows():
            self.labels_map[row["Frame_Number"]] = row["Label"]
        # Filter out image files without a corresponding label
        self.image_files = [
            img
            for img in self.image_files
            if os.path.splitext(img)[0] in self.labels_map
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        full_img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(full_img_path).convert("RGB")

        frame_identifier = os.path.splitext(img_name)[0]
        label = self.labels_map.get(frame_identifier)

        # Handle the unlikely case where a label is not found
        if label is None:
            print(f"Warning: Label not found for image: {img_name}. Skipping...")
            return None  # This should be handled by your dataloader or skipped

        if self.transform:
            image = self.transform(image)

        return image, label

# %%
# Update your DataLoader to skip None types (which we use for missing labels)
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


train_data_path = "../../data/raw/Food/train"
test_data_path = "../../data/raw/Food/test"
val_data_path = "../../data/raw/Food/val"

train_dataset = CustomImageDataset(train_data_path, labels_df, transform)
test_dataset = CustomImageDataset(
    test_data_path, labels_df, transform
)  # Adjust these according to actual splits
val_dataset = CustomImageDataset(
    val_data_path, labels_df, transform
)  # Adjust these according to actual splits

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
# !pip install efficientnet_pytorch
# !pip install optuna

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet  # Corrected import for your requirement
import optuna

# EarlyStopping class definition
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Check for GPU availability and use it if possible
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def train_model(trial, train_loader, val_loader, num_epochs=5):
    # Sample hyperparameters from the trial
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 0.1)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "Adamax", "Adagrad", "Adadelta"])
    num_epochs = trial.suggest_int("num_epochs", 500, 1500)

    # Initialize and move the model to the specified device
    model = EfficientNet.from_name("efficientnet-b0", num_classes=2).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer based on the sampled name
    optimizer = {
        "Adam": optim.Adam(model.parameters(), lr=learning_rate),
        "SGD": optim.SGD(model.parameters(), lr=learning_rate),
        "RMSprop": optim.RMSprop(model.parameters(), lr=learning_rate),
        "Adamax": optim.Adamax(model.parameters(), lr=learning_rate),
        "Adagrad": optim.Adagrad(model.parameters(), lr=learning_rate),
        "Adadelta": optim.Adadelta(model.parameters(), lr=learning_rate)
    }[optimizer_name]

    # Early stopping initialization
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Lists to store training and validation losses
    train_losses = []
    valid_losses = []

    # Train the model
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to the device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for the epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluate the model on the validation set
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        # Calculate average validation loss for the epoch
        valid_loss /= len(val_loader)
        valid_losses.append(valid_loss)

        # Report intermediate results to Optuna
        trial.report(valid_loss, epoch)

        # Early stopping call
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Print training statistics
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

    return valid_losses[-1]

# Define the objective function for Optuna
def objective(trial):
    # You can pass additional arguments to `train_model` if needed
    return train_model(trial, train_loader, val_loader)

# Create a study object and optimize hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=17)

# Get the best hyperparameters and train the final model with them
best_params = study.best_params
print("Best hyperparameters:", best_params)



# %%
import pickle

# Assuming 'study' is your Optuna study object
with open("study.pkl", "wb") as f:
    pickle.dump(study, f)

# %%



