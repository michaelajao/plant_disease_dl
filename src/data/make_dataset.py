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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from zipfile import ZipFile

# Unzipping the dataset
# with ZipFile("../../new-plant-diseases-dataset.zip", "r") as zip_ref:
#     zip_ref.extractall("../../data/raw")
    
# with ZipFile("../../plant-village-dataset-updated.zip", "r") as zip_ref:
#     zip_ref.extractall("../../data/raw/plant-village-dataset-updated")
    

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


# class PlantDiseaseDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         Custom dataset for plant disease classification that handles
#         - Crop Type Classification
#         - Disease Detection (Healthy vs Diseased)
#         - Disease Type Classification
#         """
#         self.data_dir = data_dir
#         self.transform = transform
#         self.data = []
#         self.labels = {"crop_type": [], "disease": [], "healthy": []}
#         self.class_to_idx = {
#             "crop_type": {},
#             "disease": {},
#             "healthy": {True: 1, False: 0},
#         }
#         self.idx_to_class = {
#             "crop_type": {},
#             "disease": {},
#             "healthy": {1: True, 0: False},
#         }

#         self._prepare_dataset()

#     def _prepare_dataset(self):
#         disease_folders = os.listdir(self.data_dir)
#         for folder_name in disease_folders:
#             folder_path = os.path.join(self.data_dir, folder_name)
#             images = os.listdir(folder_path)
#             plant, disease = folder_name.split("___")

#             if plant not in self.class_to_idx["crop_type"]:
#                 self.class_to_idx["crop_type"][plant] = len(
#                     self.class_to_idx["crop_type"]
#                 )
#                 self.idx_to_class["crop_type"][
#                     len(self.idx_to_class["crop_type"])
#                 ] = plant

#             if disease not in self.class_to_idx["disease"]:
#                 self.class_to_idx["disease"][disease] = len(
#                     self.class_to_idx["disease"]
#                 )
#                 self.idx_to_class["disease"][
#                     len(self.idx_to_class["disease"])
#                 ] = disease

#             for img in images:
#                 img_path = os.path.join(folder_path, img)
#                 self.data.append(img_path)
#                 self.labels["crop_type"].append(self.class_to_idx["crop_type"][plant])
#                 self.labels["disease"].append(self.class_to_idx["disease"][disease])
#                 self.labels["healthy"].append(1 if disease == "Healthy" else 0)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data[idx]
#         img = datasets.folder.default_loader(
#             img_path
#         )  # Default loader handles image opening and conversion to RGB
#         if self.transform:
#             img = self.transform(img)

#         labels = {
#             "crop_type": torch.tensor(self.labels["crop_type"][idx]),
#             "disease": torch.tensor(self.labels["disease"][idx]),
#             "healthy": torch.tensor(self.labels["healthy"][idx], dtype=torch.float),
#         }

#         return img, labels


# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )

# train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
# valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)

# # Example DataLoader setup
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32)

# # To access the mapping back from indices to class names
# print(train_dataset.idx_to_class)


# class MultiTaskCNN(nn.Module):
#     def __init__(self, num_crop_types, num_diseases):
#         super(MultiTaskCNN, self).__init__()

#         # Load a pre-trained model as feature extractor
#         # Here, we use ResNet50, but you can choose a different model like EfficientNet
#         self.feature_extractor = models.resnet50(pretrained=True)

#         # Remove the last layer (fully connected layer) of the feature extractor
#         num_features = self.feature_extractor.fc.in_features
#         self.feature_extractor.fc = nn.Identity()

#         # Task-specific layers
#         # Crop classification head
#         self.crop_head = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_crop_types),
#         )

#         # Disease detection head (binary classification: healthy vs diseased)
#         self.disease_detection_head = nn.Sequential(
#             nn.Linear(num_features, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#         # Disease classification head
#         self.disease_classification_head = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_diseases),
#         )

#     def forward(self, x):
#         # Shared feature extraction
#         x = self.feature_extractor(x)

#         # Task-specific predictions
#         crop_pred = self.crop_head(x)
#         disease_detection_pred = self.disease_detection_head(x)
#         disease_classification_pred = self.disease_classification_head(x)

#         return crop_pred, disease_detection_pred, disease_classification_pred


# class MultiTaskLoss(nn.Module):
#     def __init__(
#         self,
#         weight_crop=1.0,
#         weight_disease_detection=1.0,
#         weight_disease_classification=1.0,
#     ):
#         super(MultiTaskLoss, self).__init__()
#         self.weight_crop = weight_crop
#         self.weight_disease_detection = weight_disease_detection
#         self.weight_disease_classification = weight_disease_classification
#         self.loss_crop = nn.CrossEntropyLoss()
#         self.loss_disease_detection = nn.BCELoss()
#         self.loss_disease_classification = nn.CrossEntropyLoss()

#     def forward(self, outputs, targets):
#         # Unpack the outputs and targets
#         crop_pred, disease_detection_pred, disease_classification_pred = outputs
#         crop_target, disease_detection_target, disease_classification_target = (
#             targets["crop_type"],
#             targets["healthy"],
#             targets["disease"],
#         )

#         # Calculate individual losses
#         loss_crop = self.loss_crop(crop_pred, crop_target)
#         loss_disease_detection = self.loss_disease_detection(
#             disease_detection_pred.view(-1), disease_detection_target
#         )
#         loss_disease_classification = self.loss_disease_classification(
#             disease_classification_pred, disease_classification_target
#         )

#         # Weighted sum of the individual losses
#         total_loss = (
#             self.weight_crop * loss_crop
#             + self.weight_disease_detection * loss_disease_detection
#             + self.weight_disease_classification * loss_disease_classification
#         )

#         return total_loss


# # create a function that times the experiments, how fast it runs on the GPU
# def time_model(model, data_loader, device):
#     model.eval()
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     for inputs, labels in data_loader:
#         inputs = inputs.to(device)
#         labels = {task: labels[task].to(device) for task in labels}
#         outputs = model(inputs)
#     end.record()
#     torch.cuda.synchronize()
#     print(f"Model took {start.elapsed_time(end):.3f} milliseconds to process the data.")        
#     return start.elapsed_time(end)  



# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_score is None:
#             self.best_score = val_loss
#         elif val_loss > self.best_score + self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = val_loss
#             self.counter = 0    # Reset the counter

# # Device configuration for training on GPU if available. two GPUs are available make use of both of them
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Model instantiation
# num_crop_types = len(plant_names)
# num_diseases = len(disease_names) + 1  # Including 'Healthy' as a type of 'disease'
# model = MultiTaskCNN(num_crop_types, num_diseases).to(device)

# # Loss function and optimizer
# criterion = MultiTaskLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # create a training loop function that trains the model on batches of data (training and validation). calculate the loss and accuracy of the model per batch. print out what is happening in the training loop
# def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=10):     
#     train_losses = []
#     val_losses = []
#     train_accuracies = []
#     val_accuracies = []
#     early_stopping = EarlyStopping(patience=5, min_delta=0.0)
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         correct = 0
#         total = 0
#         for inputs, labels in tqdm(train_loader):
#             inputs = inputs.to(device)
#             labels = {task: labels[task].to(device) for task in labels}

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#             crop_pred, _, _ = outputs
#             _, crop_pred = crop_pred.max(1)
#             correct += crop_pred.eq(labels["crop_type"]).sum().item()
#             total += inputs.size(0)

#         train_losses.append(train_loss / len(train_loader))
#         train_accuracy = correct / total
#         train_accuracies.append(train_accuracy)

#         val_loss = 0.0
#         correct = 0
#         total = 0
#         model.eval()
#         with torch.no_grad():
#             for inputs, labels in valid_loader:
#                 inputs = inputs.to(device)
#                 labels = {task: labels[task].to(device) for task in labels}

#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()

#                 crop_pred, _, _ = outputs
#                 _, crop_pred = crop_pred.max(1)
#                 correct += crop_pred.eq(labels["crop_type"]).sum().item()
#                 total += inputs.size(0)

#         val_losses.append(val_loss / len(valid_loader))
#         val_accuracy = correct / total
#         val_accuracies.append(val_accuracy)

#         print(
#             f"Epoch {epoch+1}/{num_epochs}, "
#             f"Train Loss: {train_loss / len(train_loader):.4f}, "
#             f"Train Accuracy: {train_accuracy:.2f}, "
#             f"Val Loss: {val_loss / len(valid_loader):.4f}, "
#             f"Val Accuracy: {val_accuracy:.2f}"
#         )

#         scheduler.step(val_loss)

#         early_stopping(val_loss)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break

#     return model, train_losses, val_losses, train_accuracies, val_accuracies


# # plot the training and validation losses
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label="Training Loss", color="skyblue")
# plt.plot(val_losses, label="Validation Loss", color="orange")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.grid(True)
# plt.savefig("../../reports/figures/losses.png")
# plt.show()

# # plot the training and validation accuracies
# plt.figure(figsize=(10, 6))
# plt.plot(train_accuracies, label="Training Accuracy", color="skyblue")
# plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")
# plt.title("Training and Validation Accuracy")
# plt.legend()
# plt.grid(True)
# plt.savefig("../../reports/figures/accuracies.png")
# plt.show()

# # evaluate the model
# def evaluate(model, data_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     predictions = []
#     targets = []

#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = {task: labels[task].to(device) for task in labels}

#             outputs = model(inputs)

#             crop_pred, _, _ = outputs
#             _, crop_pred = crop_pred.max(1)
#             correct += crop_pred.eq(labels["crop_type"]).sum().item()
#             total += inputs.size(0)

#             predictions.extend(crop_pred.tolist())
#             targets.extend(labels["crop_type"].tolist())

#     accuracy = correct / total
#     return accuracy, predictions, targets


# train_accuracy, train_predictions, train_targets = evaluate(model, train_loader, device)

# valid_accuracy, valid_predictions, valid_targets = evaluate(model, valid_loader, device)

# print(f"Training Accuracy: {train_accuracy:.2f}")
# print(f"Validation Accuracy: {valid_accuracy:.2f}")

# # Calculate additional evaluation metrics
# train_precision = precision_score(train_targets, train_predictions, average="macro")
# train_recall = recall_score(train_targets, train_predictions, average="macro")
# train_f1 = f1_score(train_targets, train_predictions, average="macro")

# valid_precision = precision_score(valid_targets, valid_predictions, average="macro")
# valid_recall = recall_score(valid_targets, valid_predictions, average="macro")
# valid_f1 = f1_score(valid_targets, valid_predictions, average="macro")

# print(
#     f"Training Precision: {train_precision:.2f}, Recall: {train_recall:.2f}, F1 Score: {train_f1:.2f}"
# )
# print(
#     f"Validation Precision: {valid_precision:.2f}, Recall: {valid_recall:.2f}, F1 Score: {valid_f1:.2f}"
# )


# # Visualizing predictions make it a plot side by side with the probability of the prediction. make the plot good enough to be saved as an image for my research paper



# # Save the model
    