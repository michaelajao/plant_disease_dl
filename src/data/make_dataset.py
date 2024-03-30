import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm


data_path = "../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Extracting disease folders from the training directory
diseases = os.listdir(train_dir)

plant_names = []
disease_names = []
healthy_labels = []

# Separating plant names and disease names, including a separate category for 'healthy'
for disease_folder in diseases:
    parts = disease_folder.split("___")
    plant = parts[0]
    disease = (
        parts[1] if len(parts) > 1 else "Healthy"
    )  # Assumes disease label is present or assigns 'Healthy'

    if plant not in plant_names:
        plant_names.append(plant)

    if disease == "Healthy":
        healthy_labels.append(
            f"{plant}___Healthy"
        )  # Keeping track of healthy labels for disease detection
    elif disease not in disease_names:
        disease_names.append(disease)

# Count the number of images for each disease in the dataset
disease_count = {}
for disease_folder in diseases:
    disease_path = os.path.join(train_dir, disease_folder)
    disease_count[disease_folder] = len(os.listdir(disease_path))

# Convert the disease_count dictionary to a pandas DataFrame for better analysis and visualization
disease_count_df = pd.DataFrame(
    disease_count.values(), index=disease_count.keys(), columns=["no_of_images"]
)

print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases (excluding healthy): {len(disease_names)}")
print(f"Total classes (including healthy labels per plant): {len(diseases)}")

# Optionally, display the DataFrame for analysis
print(disease_count_df)

# Create a dataframe for counting how many images are available for each crop and disease in the training and validation
diseases = os.listdir(train_dir)
train_data = []
valid_data = []

for disease in diseases:
    train_data.append(
        {
            "plant": disease.split("___")[0],
            "disease": disease.split("___")[1],
            "no_of_images": len(os.listdir(os.path.join(train_dir, disease))),
        }
    )

    valid_data.append(
        {
            "plant": disease.split("___")[0],
            "disease": disease.split("___")[1],
            "no_of_images": len(os.listdir(os.path.join(valid_dir, disease))),
        }
    )

train_data = pd.DataFrame(train_data)
valid_data = pd.DataFrame(valid_data)

train_data = train_data.groupby(["plant", "disease"]).sum().reset_index()
valid_data = valid_data.groupby(["plant", "disease"]).sum().reset_index()

train_data["data"] = "train"
valid_data["data"] = "valid"

data = pd.concat([train_data, valid_data])

data = data.pivot(
    index=["plant", "disease"], columns="data", values="no_of_images"
).reset_index()

data = data.fillna(0)

data["total_images"] = data.train + data.valid

data = data.sort_values(by="total_images", ascending=False)

data = data.reset_index(drop=True)

data.to_csv("../../data/processed/data.csv", index=False)


class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for plant disease classification that handles
        - Crop Type Classification
        - Disease Detection (Healthy vs Diseased)
        - Disease Type Classification
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = {"crop_type": [], "disease": [], "healthy": []}
        self.class_to_idx = {
            "crop_type": {},
            "disease": {},
            "healthy": {True: 1, False: 0},
        }
        self.idx_to_class = {
            "crop_type": {},
            "disease": {},
            "healthy": {1: True, 0: False},
        }

        self._prepare_dataset()

    def _prepare_dataset(self):
        disease_folders = os.listdir(self.data_dir)
        for folder_name in disease_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            images = os.listdir(folder_path)
            plant, disease = folder_name.split("___")

            if plant not in self.class_to_idx["crop_type"]:
                self.class_to_idx["crop_type"][plant] = len(
                    self.class_to_idx["crop_type"]
                )
                self.idx_to_class["crop_type"][
                    len(self.idx_to_class["crop_type"])
                ] = plant

            if disease not in self.class_to_idx["disease"]:
                self.class_to_idx["disease"][disease] = len(
                    self.class_to_idx["disease"]
                )
                self.idx_to_class["disease"][
                    len(self.idx_to_class["disease"])
                ] = disease

            for img in images:
                img_path = os.path.join(folder_path, img)
                self.data.append(img_path)
                self.labels["crop_type"].append(self.class_to_idx["crop_type"][plant])
                self.labels["disease"].append(self.class_to_idx["disease"][disease])
                self.labels["healthy"].append(1 if disease == "Healthy" else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = datasets.folder.default_loader(
            img_path
        )  # Default loader handles image opening and conversion to RGB
        if self.transform:
            img = self.transform(img)

        labels = {
            "crop_type": torch.tensor(self.labels["crop_type"][idx]),
            "disease": torch.tensor(self.labels["disease"][idx]),
            "healthy": torch.tensor(self.labels["healthy"][idx], dtype=torch.float),
        }

        return img, labels


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

# Example DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# To access the mapping back from indices to class names
print(train_dataset.idx_to_class)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crop_types, num_diseases):
        super(MultiTaskCNN, self).__init__()

        # Load a pre-trained model as feature extractor
        # Here, we use ResNet50, but you can choose a different model like EfficientNet
        self.feature_extractor = models.resnet50(pretrained=True)

        # Remove the last layer (fully connected layer) of the feature extractor
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        # Task-specific layers
        # Crop classification head
        self.crop_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_crop_types),
        )

        # Disease detection head (binary classification: healthy vs diseased)
        self.disease_detection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Disease classification head
        self.disease_classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        # Shared feature extraction
        x = self.feature_extractor(x)

        # Task-specific predictions
        crop_pred = self.crop_head(x)
        disease_detection_pred = self.disease_detection_head(x)
        disease_classification_pred = self.disease_classification_head(x)

        return crop_pred, disease_detection_pred, disease_classification_pred


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_crop=1.0,
        weight_disease_detection=1.0,
        weight_disease_classification=1.0,
    ):
        super(MultiTaskLoss, self).__init__()
        self.weight_crop = weight_crop
        self.weight_disease_detection = weight_disease_detection
        self.weight_disease_classification = weight_disease_classification
        self.loss_crop = nn.CrossEntropyLoss()
        self.loss_disease_detection = nn.BCELoss()
        self.loss_disease_classification = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Unpack the outputs and targets
        crop_pred, disease_detection_pred, disease_classification_pred = outputs
        crop_target, disease_detection_target, disease_classification_target = (
            targets["crop_type"],
            targets["healthy"],
            targets["disease"],
        )

        # Calculate individual losses
        loss_crop = self.loss_crop(crop_pred, crop_target)
        loss_disease_detection = self.loss_disease_detection(
            disease_detection_pred.view(-1), disease_detection_target
        )
        loss_disease_classification = self.loss_disease_classification(
            disease_classification_pred, disease_classification_target
        )

        # Weighted sum of the individual losses
        total_loss = (
            self.weight_crop * loss_crop
            + self.weight_disease_detection * loss_disease_detection
            + self.weight_disease_classification * loss_disease_classification
        )

        return total_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Device configuration for training on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
num_crop_types = len(plant_names)
num_diseases = len(disease_names) + 1  # Including 'Healthy' as a type of 'disease'
model = MultiTaskCNN(num_crop_types, num_diseases).to(device)

# Loss function and optimizer
criterion = MultiTaskLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    early_stopping,
    num_epochs=25,
):
    # Initialize lists to track per epoch losses
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training phase with tqdm progress bar
        with tqdm(train_loader, unit="batch") as train_epoch:
            for inputs, labels in train_epoch:
                train_epoch.set_description(f"Epoch {epoch+1}/{num_epochs} [Train]")

                inputs = inputs.to(device)
                labels = {task: labels[task].to(device) for task in labels}

                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize

                running_loss += loss.item() * inputs.size(0)
                train_epoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase with tqdm progress bar
        model.eval()  # Set model to evaluate mode
        val_running_loss = 0.0
        with torch.no_grad(), tqdm(valid_loader, unit="batch") as valid_epoch:
            for inputs, labels in valid_epoch:
                valid_epoch.set_description(f"Epoch {epoch+1}/{num_epochs} [Validate]")

                inputs = inputs.to(device)
                labels = {task: labels[task].to(device) for task in labels}

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                valid_epoch.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / len(valid_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}"
        )

        # Early stopping check
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    return train_losses, val_losses


# Train the model
early_stopping = EarlyStopping(patience=5, min_delta=0.01)
train_losses, val_losses = train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    early_stopping,
    num_epochs=25,
)
