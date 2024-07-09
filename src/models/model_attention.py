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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

plt.rcParams.update({
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
})

data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

diseases = os.listdir(train_dir)
plant_names, disease_names, healthy_labels = [], [], []

for disease_folder in diseases:
    parts = disease_folder.split("___")
    plant = parts[0]
    disease = parts[1] if len(parts) > 1 else "Healthy"

    if plant not in plant_names:
        plant_names.append(plant)
    if disease == "Healthy":
        healthy_labels.append(f"{plant}___Healthy")
    elif disease not in disease_names:
        disease_names.append(disease)

disease_count = {disease_folder: len(os.listdir(os.path.join(train_dir, disease_folder))) for disease_folder in diseases}
disease_count_df = pd.DataFrame(disease_count.values(), index=disease_count.keys(), columns=["no_of_images"])

print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases (excluding healthy): {len(disease_names)}")
print(f"Total classes (including healthy labels per plant): {len(diseases)}")
print(f"Total number of images: {sum(disease_count.values())}")
print(disease_count_df)

train_data, valid_data = [], []

for disease in diseases:
    train_data.append({"plant": disease.split("___")[0], "disease": disease.split("___")[1], "no_of_images": len(os.listdir(os.path.join(train_dir, disease)))})
    valid_data.append({"plant": disease.split("___")[0], "disease": disease.split("___")[1], "no_of_images": len(os.listdir(os.path.join(valid_dir, disease)))})

train_data, valid_data = pd.DataFrame(train_data), pd.DataFrame(valid_data)
train_data = train_data.groupby(["plant", "disease"]).sum().reset_index()
valid_data = valid_data.groupby(["plant", "disease"]).sum().reset_index()
train_data["data"], valid_data["data"] = "train", "valid"
data = pd.concat([train_data, valid_data])
data = data.pivot(index=["plant", "disease"], columns="data", values="no_of_images").reset_index().fillna(0)
data["total_images"] = data.train + data.valid
data = data.sort_values(by="total_images", ascending=False).reset_index(drop=True)
data.to_csv("../../data/processed/data.csv", index=False)

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data, self.labels = [], {"crop_type": [], "disease": [], "healthy": []}
        self.class_to_idx = {"crop_type": {}, "disease": {}, "healthy": {True: 1, False: 0}}
        self.idx_to_class = {"crop_type": {}, "disease": {}, "healthy": {1: True, 0: False}}
        self._prepare_dataset()

    def _prepare_dataset(self):
        disease_folders = os.listdir(self.data_dir)
        for folder_name in disease_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            images = os.listdir(folder_path)
            plant, disease = folder_name.split("___")

            if plant not in self.class_to_idx["crop_type"]:
                self.class_to_idx["crop_type"][plant] = len(self.class_to_idx["crop_type"])
                self.idx_to_class["crop_type"][len(self.idx_to_class["crop_type"])] = plant

            if disease not in self.class_to_idx["disease"]:
                self.class_to_idx["disease"][disease] = len(self.class_to_idx["disease"])
                self.idx_to_class["disease"][len(self.idx_to_class["disease"])] = disease

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
        img = datasets.folder.default_loader(img_path)
        if self.transform:
            img = self.transform(img)

        labels = {
            "crop_type": torch.tensor(self.labels["crop_type"][idx]),
            "disease": torch.tensor(self.labels["disease"][idx]),
            "healthy": torch.tensor(self.labels["healthy"][idx], dtype=torch.float),
        }

        return img, labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = PlantDiseaseDataset(train_dir, transform=transform)
valid_dataset = PlantDiseaseDataset(valid_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

class EfficientNetB0MultiTask(nn.Module):
    def __init__(self, num_crop_types, num_diseases):
        super(EfficientNetB0MultiTask, self).__init__()
        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        num_features = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()
        
        self.crop_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_crop_types),
        )
        self.disease_detection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.disease_classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        crop_pred = self.crop_head(x)
        disease_detection_pred = self.disease_detection_head(x)
        disease_classification_pred = self.disease_classification_head(x)
        return crop_pred, disease_detection_pred, disease_classification_pred

class MultiTaskLoss(nn.Module):
    def __init__(self, weight_crop=1.0, weight_disease_detection=1.0, weight_disease_classification=1.0):
        super(MultiTaskLoss, self).__init__()
        self.weight_crop = weight_crop
        self.weight_disease_detection = weight_disease_detection
        self.weight_disease_classification = weight_disease_classification
        self.loss_crop = nn.CrossEntropyLoss()
        self.loss_disease_detection = nn.BCELoss()
        self.loss_disease_classification = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        crop_pred, disease_detection_pred, disease_classification_pred = outputs
        crop_target, disease_detection_target, disease_classification_target = targets["crop_type"], targets["healthy"], targets["disease"]
        loss_crop = self.loss_crop(crop_pred, crop_target)
        loss_disease_detection = self.loss_disease_detection(disease_detection_pred.view(-1), disease_detection_target)
        loss_disease_classification = self.loss_disease_classification(disease_classification_pred, disease_classification_target)
        total_loss = (self.weight_crop * loss_crop + self.weight_disease_detection * loss_disease_detection + self.weight_disease_classification * loss_disease_classification)
        return total_loss

def evaluate(model, valid_loader, device):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            outputs = model(images)
            crop_pred, disease_detection_pred, disease_classification_pred = outputs
            disease_detection_pred = (disease_detection_pred > 0.5).float()
            predictions.extend(disease_classification_pred.argmax(dim=1).cpu().numpy())
            targets.extend(labels["disease"].cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    roc_auc = roc_auc_score(targets, predictions, average='weighted', multi_class='ovo')
    precision, recall, _ = precision_recall_curve(targets, predictions)
    return accuracy, roc_auc, precision, recall, predictions, targets

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
            self.counter = 0

torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_crop_types = len(plant_names)
num_diseases = len(disease_names) + 1  # Including 'Healthy' as a type of 'disease'
model = EfficientNetB0MultiTask(num_crop_types, num_diseases).to(device)
criterion = MultiTaskLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

writer = SummaryWriter()

def training_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=3):
    train_losses, valid_losses, valid_accuracies = [], [], []
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
        accuracy, roc_auc, precision, recall, _, _ = evaluate(model, valid_loader, device)
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
            f"Valid ROC-AUC: {roc_auc:.4f}, "
            f"Time: {time.time() - start_time:.2f}s"
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', accuracy, epoch)
        writer.add_scalar('ROC-AUC/valid', roc_auc, epoch)
        writer.add_scalar('Precision/valid', precision.mean(), epoch)
        writer.add_scalar('Recall/valid', recall.mean(), epoch)

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()
    return train_losses, valid_losses, valid_accuracies

train_losses, valid_losses, valid_accuracies = training_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=5)

def plot_learning_curves(train_losses, valid_losses, valid_accuracies):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../reports/figures/training_results_multitask_problem.pdf")
    plt.show()

plot_learning_curves(train_losses, valid_losses, valid_accuracies)

torch.save(model.state_dict(), "../../models/multitask_model.pth")
model.load_state_dict(torch.load("../../models/multitask_model.pth"))

_, predictions, targets = evaluate(model, valid_loader, device)
cm = confusion_matrix(targets, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("../../reports/figures/confusion_matrix_multitask_problem.pdf")
plt.show()

precision = precision_score(targets, predictions, average="weighted")
recall = recall_score(targets, predictions, average="weighted")
f1 = f1_score(targets, predictions, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

def plot_feature_maps(model, image, layer_num):
    model.eval()
    model.to(device)
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
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    for i in range(64):
        ax = axes[i // 8, i % 8]
        ax.imshow(feature_maps[i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("../../reports/figures/feature_maps_multitask_problem.pdf")
    plt.show()

img, _ = valid_dataset[0]
plot_feature_maps(model, img, 0)
plot_feature_maps(model, img, 7)

fig, axes = plt.subplots(2, 5, figsize=(20, 15))
for i in range(10):
    idx = np.random.randint(len(valid_dataset))
    img, labels = valid_dataset[idx]
    img = img.unsqueeze(0)
    with torch.no_grad():
        crop_pred, disease_detection_pred, disease_classification_pred = model(img.to(device))
    crop_pred_prob = torch.softmax(crop_pred, dim=1)
    crop_pred_label = crop_pred_prob.argmax(dim=1).item()
    crop_pred_prob = crop_pred_prob.squeeze().tolist()
    disease_detection_prob = torch.sigmoid(disease_detection_pred).item()
    disease_detection_label = "Healthy" if disease_detection_prob > 0.5 else "Diseased"
    disease_classification_prob = torch.softmax(disease_classification_pred, dim=1)
    disease_classification_label = disease_names[disease_classification_prob.argmax(dim=1).item()]
    disease_classification_prob = disease_classification_prob.squeeze().tolist()
    plant = valid_dataset.idx_to_class["crop_type"][labels["crop_type"].item()]
    disease = valid_dataset.idx_to_class["disease"][labels["disease"].item()]
    healthy = valid_dataset.idx_to_class["healthy"][labels["healthy"].item()]
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
