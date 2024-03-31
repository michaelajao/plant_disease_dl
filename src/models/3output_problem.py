import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
print(f"Total number of images: {sum(disease_count.values())}")
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
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# To access the mapping back from indices to class names
print(train_dataset.idx_to_class)

# Plot the random 10 images across crop, disease, and healthy / unhealthy labels
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(10):
    idx = np.random.randint(len(train_dataset))
    img, labels = train_dataset[idx]
    plant = train_dataset.idx_to_class["crop_type"][labels["crop_type"].item()]
    disease = train_dataset.idx_to_class["disease"][
        labels["disease"].item()
    ]  # Skip 'Healthy' label
    healthy = train_dataset.idx_to_class["healthy"][
        labels["healthy"].item()
    ]  # Convert binary label to 'Healthy' or 'Diseased'

    ax = axes[i // 5, i % 5]
    ax.imshow(img.permute(1, 2, 0))
    ax.axis("off")
    ax.set_title(f"{plant}\n{disease}\n{healthy}")

plt.tight_layout()
plt.savefig("../../reports/figures/sample_images_multitask_problem.png")
plt.show()


# Define a multi-task CNN model that predicts the crop type, disease detection, and disease classification
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


# Define a multi-task loss function that combines the losses for crop classification, disease detection, and disease classification
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


# evaluate the model on the validation data and print out the accuracy of the model on the validation data
def evaluate(model, valid_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}

            outputs = model(images)
            crop_pred, disease_detection_pred, disease_classification_pred = outputs

            # Disease detection predictions are probabilities, convert to binary predictions
            disease_detection_pred = (disease_detection_pred > 0.5).float()

            predictions.extend(disease_classification_pred.argmax(dim=1).cpu().numpy())
            targets.extend(labels["disease"].cpu().numpy())

    accuracy = accuracy_score(targets, predictions)

    return accuracy, predictions, targets


# Define an early stopping class that stops training if the validation loss does not decrease after a certain number of epochs
class EarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0  # Reset the counter if the validation loss decreases


# Device configuration for training on GPU if available. two GPUs are available make use of both of them
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
num_crop_types = len(plant_names)
num_diseases = len(disease_names) + 1  # Including 'Healthy' as a type of 'disease'
model = MultiTaskCNN(num_crop_types, num_diseases).to(device)

# Loss function and optimizer
criterion = MultiTaskLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# create a training loop that trains the model on batches of data (training and validation). calculate the loss and accuracy of the model per batch using the validation data. print out what is happening for every 500 batches in the training loop and time the experiment to see how long it takes to train the model on the GPU
def training_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
):
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # Time the experiment

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n{'-' * 10}")

        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:
                print(f"\nBatch {i + 1}, Loss: {running_loss / 500}")
                running_loss = 0.0

        # Evaluate the model on the validation data and print the accuracy
        valid_accuracy, _, _ = evaluate(model, valid_loader, device)
        print(f"\nValidation Accuracy: {valid_accuracy:.2f}")

        # Save the model if the validation accuracy has increased
        if not early_stopping(valid_accuracy):
            torch.save(model.state_dict(), "../../models/model.pth")
            print("Model saved!\n")

        train_losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_accuracy)

        scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    end_time = time.time()

    print(f"Training time: {end_time - start_time:.0f}s")

    return train_losses, valid_losses


train_losses, valid_losses = training_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
)

# Plot the training and validation losses
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("../../reports/figures/training_validation_losses_3_output.png")
plt.show()
