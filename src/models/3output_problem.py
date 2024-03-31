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

plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 14,
        "figure.figsize": [15, 8],
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.75",
        "legend.fontsize": "medium",
        "legend.fancybox": False,
        "legend.frameon": False,
        "legend.shadow": False,
        "savefig.transparent": True,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.dpi": 400,
        
    }
)


data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
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

        Args:
            data_dir (str): The directory path where the dataset is located.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default: None.
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
        """
        Private method to prepare the dataset by populating the data and labels lists.
        """
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
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the sample and its corresponding labels at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding labels.
        """
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
    ax.set_title(f"Plant: {plant}\nDisease: {disease}\nHealthy: {healthy}")

plt.tight_layout()
plt.savefig("../../reports/figures/sample_images_multitask_problem.pdf")
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instantiation
num_crop_types = len(plant_names)
num_diseases = len(disease_names) + 1  # Including 'Healthy' as a type of 'disease'
model = MultiTaskCNN(num_crop_types, num_diseases).to(device)

# Loss function and optimizer
criterion = MultiTaskLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# create a training loop function that trains the model on training and validation. calculate the loss and accuracy of the model per batch using the validation data. print out what is happening in the training loop and time the experiment to see how long it takes to train the model on the GPU


def training_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
):
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n{'-' * 20}")
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader):
    
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        valid_loss = 0.0
        accuracy, _, _ = evaluate(model, valid_loader, device)
        valid_accuracies.append(accuracy)

        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = {key: value.to(device) for key, value in labels.items()}

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Valid Accuracy: {accuracy:.4f}, "
            f"Time: {time.time() - start_time:.2f}s, "
            f"Time in minutes: {(time.time() - start_time) / 60}m, "
            f"LR: {scheduler.get_last_lr()}"
        )

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, valid_losses, valid_accuracies


# Train the model
train_losses, valid_losses, valid_accuracies = training_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
)

# Plot the training and validation losses and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(valid_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("../../reports/figures/training_results_multitask_problem.pdf")
plt.show()

# save the trained model
torch.save(model.state_dict(), "../../models/multitask_model.pth")

# load the trained model

model.load_state_dict(torch.load("../../models/multitask_model.pth"))

# create a confusion matrix to see how well the model is performing on the validation data
from sklearn.metrics import confusion_matrix
import seaborn as sns

_, predictions, targets = evaluate(model, valid_loader, device)
cm = confusion_matrix(targets, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("../../reports/figures/confusion_matrix_multitask_problem.pdf")
plt.show()

# calculate the precision, recall, and f1 score of the model on the validation data
precision = precision_score(targets, predictions, average="weighted")
recall = recall_score(targets, predictions, average="weighted")
f1 = f1_score(targets, predictions, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# plot the feature maps of the model to see what the model is learning
def plot_feature_maps(model, image, layer_num):
    model.eval()
    model.to(device)

    # Extract the feature maps
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)

    layer = model.feature_extractor[layer_num]
    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        model(image)

    hook.remove()

    feature_maps = feature_maps[0].squeeze().cpu()

    # Plot the feature maps
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    for i in range(64):
        ax = axes[i // 8, i % 8]
        ax.imshow(feature_maps[i], cmap="viridis")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("../../reports/figures/feature_maps_multitask_problem.pdf")
    plt.show()
    
# Plot the feature maps of the first convolutional layer
img, _ = valid_dataset[0]
plot_feature_maps(model, img, 0)

# Plot the feature maps of the last convolutional layer
img, _ = valid_dataset[0]
plot_feature_maps(model, img, 7)





# Plot the results of the model on the validation data for 10 random images in the test data
fig, axes = plt.subplots(2, 5, figsize=(20, 15))
for i in range(10):
    idx = np.random.randint(len(valid_dataset))
    img, labels = valid_dataset[idx]
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        crop_pred, disease_detection_pred, disease_classification_pred = model(
            img.to(device)
        )

    crop_pred_prob = torch.softmax(crop_pred, dim=1)
    crop_pred_label = crop_pred_prob.argmax(dim=1).item()
    crop_pred_prob = crop_pred_prob.squeeze().tolist()

    disease_detection_prob = torch.sigmoid(disease_detection_pred).item()
    disease_detection_label = "Healthy" if disease_detection_prob > 0.5 else "Diseased"

    disease_classification_prob = torch.softmax(disease_classification_pred, dim=1)
    disease_classification_label = disease_names[
        disease_classification_prob.argmax(dim=1).item()
    ]
    disease_classification_prob = disease_classification_prob.squeeze().tolist()

    plant = valid_dataset.idx_to_class["crop_type"][labels["crop_type"].item()]
    disease = valid_dataset.idx_to_class["disease"][
        labels["disease"].item()
    ]  # Skip 'Healthy' label
    healthy = valid_dataset.idx_to_class["healthy"][
        labels["healthy"].item()
    ]  # Convert binary label to 'Healthy' or 'Diseased'

    ax = axes[i // 5, i % 5]
    ax.imshow(img.squeeze().permute(1, 2, 0))
    ax.axis("off")
    ax.set_title(
        f"Plant: {plant}\n"
        f"Disease: {disease}\n"
        f"Healthy: {healthy}\n"
        f"Predicted Crop: {plant_names[crop_pred_label]} (prob % {crop_pred_prob[crop_pred_label] * 100:.2f})\n"
        f"Predicted Disease Detection: {disease_detection_label} (prob % {disease_detection_prob * 100:.2f})\n"
        f"Predicted Disease: {disease_classification_label} (prob % {disease_classification_prob[disease_classification_pred.argmax(dim=1).item()] * 100:.2f})"
    )

plt.tight_layout()
plt.savefig("../../reports/figures/prediction_results_multitask_problem.pdf")
plt.show()
