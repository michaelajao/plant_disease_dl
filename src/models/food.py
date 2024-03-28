import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import torch

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

import kaggle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("PyTorch version:", torch.__version__)

torch.manual_seed(100)
np.random.seed(100)

# downloading the data from kaggle using the kaggle api environment variable
# !pip install kaggle
# !mkdir ~/.kaggle
# !echo '{"username":"kaggle_username","key":"kaggle_key"}' > ~/.kaggle/kaggle.json
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d kmader/food41
{"username":"michaelajaoolarinoye","key":"bfd2e06a6264c3ea8c1951de478ab648"}
# !unzip food41.zip -d ../../data/raw




# print using GPU if available and how many
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the train , test and validation data and labels
print(os.listdir("../../data/raw/train"))
labels_df = pd.read_csv("../../data/raw/labels/labels.csv")
# Define the data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


# Update your DataLoader to skip None types (which we use for missing labels)
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


train_data_path = "../../data/raw/train"
test_data_path = "../../data/raw/test"
val_data_path = "../../data/raw/val"

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


class CustomCNN(nn.Module):
    def __init__(self, num_layers, hidden_units, num_classes=10):
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = 3  # Assuming input images are RGB

        for i in range(num_layers):
            out_channels = hidden_units[i]
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(
            in_channels * (224 // 2**num_layers) ** 2, num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


num_layers = 3
hidden_units = [32, 64, 128]  # Number of channels for each layer
num_classes = 2  # Adjust based on your dataset
model = CustomCNN(num_layers, hidden_units, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.Adam(model.parameters(), lr=0.001)

# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import numpy as np

# def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
#     losses = [[], []]
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for i, data in enumerate(tqdm(train_loader)):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         losses[0].append(running_loss / len(train_loader))

#         # Evaluate the model on the validation set
#         model.eval()
#         val_running_loss = 0.0
#         for i, data in enumerate(val_loader):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             val_running_loss += loss.item()

#         losses[1].append(val_running_loss / len(val_loader))

#         print(
#             f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_running_loss/len(val_loader)}"
#         )

#     print("Finished Training")
#     return losses



def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3):
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": running_loss / (len(progress_bar) + 1)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    return train_losses, val_losses


# Train the model
num_epochs = 100
losses = train(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs)

# PLotting the losses
plt.figure(figsize=(12, 6))
plt.plot(losses[0], label="Train Loss")
plt.plot(losses[1], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the losses
plot_losses(losses[0], losses[1])

# check the accuracy of the model
def accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


train_accuracy = accuracy(model, train_loader)
test_accuracy = accuracy(model, test_loader)
val_accuracy = accuracy(model, val_loader)

# check the confusion matrix
def confusion_matrix(model, data_loader):
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

test_confusion_matrix = confusion_matrix(model, test_loader)

#plot the confusion matrix

plt.figure(figsize=(10, 8))
sns.heatmap(test_confusion_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

