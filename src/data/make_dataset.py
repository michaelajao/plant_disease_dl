from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(10)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matplotlib configurations for better visualization aesthetics
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
    }
)


print(
    os.listdir(
        "../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    )
)

print(
    len(
        os.listdir(
            "../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
        )
    )
)

data_path = "../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

diseases = os.listdir(train_dir)

plant_names = []
for disease in diseases:
    plant = disease.split("___")[0]
    if plant not in plant_names:
        plant_names.append(plant)

disease_names = []
for disease in diseases:
    disease = disease.split("___")[1]
    if disease not in disease_names:
        disease_names.append(disease)

print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases: {len(disease_names)}")

disease_count = {}
for disease in diseases:
    if disease not in disease_count:
        disease_count[disease] = len(os.listdir(os.path.join(train_dir, disease)))

disease_count = pd.DataFrame(
    disease_count.values(), index=disease_count.keys(), columns=["no_of_images"]
)

disease_count.sort_values(by="no_of_images", ascending=False).plot(
    kind="bar", figsize=(15, 8), color="skyblue"
)
plt.title("Number of images for each disease")
plt.xlabel("Disease")
plt.ylabel("Number of images")
plt.show()

print(
    f"Total number of images available for training: {disease_count.no_of_images.sum()}"
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

print(f"Number of classes: {len(train_data.classes)}")

image, label = train_data[200]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
plt.imshow(image.permute(1, 2, 0))
plt.title(train_data.classes[label])
plt.show()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


show_batch(train_loader)


# Model definition
class CustomCNN(nn.Module):
    def __init__(self, num_layers, hidden_units, num_classes):
        super(CustomCNN, self).__init__()
        self.features = self._make_layers(num_layers, hidden_units)
        in_features = hidden_units[-1] * (224 // 2**num_layers) ** 2
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, num_layers, hidden_units):
        layers = []
        in_channels = 3
        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_channels, hidden_units[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_units[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = hidden_units[i]
        return nn.Sequential(*layers)


# Model instantiation
num_classes = len(train_data.classes)
model = CustomCNN(num_layers=3, hidden_units=[32, 64, 128], num_classes=num_classes).to(
    device
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Early stopping
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


# Training loop


def train(
    model, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs, device
):
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

        valid_loss /= len(valid_loader.dataset)

        print(
            f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}"
        )

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    return model


model = train(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    n_epochs=50,
    device=device,
)


# Visualize training and validation loss
def get_losses(model, train_loader, valid_loader, criterion, device):
    train_loss = 0.0
    valid_loss
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item() * images.size(0)
    valid_loss /= len(valid_loader.dataset)

    return train_loss, valid_loss


losses = get_losses(model, train_loader, valid_loader, criterion, device)


plt.figure(figsize=(12, 6))
plt.plot(losses[0], label="Train Loss")
plt.plot(losses[1], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


def evaluate_model_performance(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )


evaluate_model_performance(model, valid_loader, device)


def predict_and_visualize(image_path, model, class_names, device):
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)

    plt.imshow(image)
    plt.title(
        f"Predicted: {class_names[predicted.item()]} ({probabilities.max().item()*100:.2f}%)"
    )
    plt.axis("off")
    plt.show()


class_names = train_data.classes
for image in os.listdir(os.path.join(data_path, "valid", "Tomato___Late_blight"))[:10]:
    predict_and_visualize(
        os.path.join(data_path, "valid", "Tomato___Late_blight", image),
        model,
        class_names,
        device,
    )

# torch.save(model.state_dict(), "../../models/plant_disease_model.pth")


model.eval()
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        evaluate_model_performance(preds, labels)
        break

test_dir = "../../data/raw/test/"
test = datasets.ImageFolder(
    test_dir,
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

test_images = os.listdir(os.path.join(test_dir, "test"))
print(f"Number of test images: {len(test_images)}")


def predict_and_visualize(image_path, model, class_names, device):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, dim=1)

    plt.imshow(image)
    plt.title(
        f"Predicted: {class_names[top_catid.item()]} - {top_prob.item()*100:.2f}%"
    )
    plt.axis("off")
    plt.show()


class_names = train_data.classes
for image in test_images[:10]:
    predict_and_visualize(
        os.path.join(test_dir, "test", image), model, class_names, device
    )

torch.save(model.state_dict(), "../../models/plant_disease_model.pth")

# import torchvision.models as models

# resnet50 = models.resnet50(pretrained=True)

# for param in resnet50.parameters():
#     param.requires_grad = False

# num_ftrs = resnet50.fc.in_features
# resnet50.fc = nn.Linear(num_ftrs, len(train_data.classes))

# resnet50 = resnet50.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# n_epochs = 50
# early_stopping = EarlyStopping(patience=3, min_delta=0.01)

# train_losses = []
# valid_losses = []

# for epoch in range(n_epochs):
#     print(f"Epoch {epoch + 1}\n-------------------------------")
#     resnet50.train()
#     train_loss = 0.0
#     valid_loss = 0.0
#     for images, labels in tqdm(train_loader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = resnet50(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * images.size(0)
#     train_loss = train_loss / len(train_loader.dataset)
#     train_losses.append(train_loss)
#     print(f"Training Loss: {train_loss}")

#     resnet50.eval()
#     with torch.no_grad():
#         for images, labels in valid_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = resnet50(images)
#             loss = criterion(outputs, labels)
#             valid_loss += loss.item() * images.size(0)
#         valid_loss = valid_loss / len(valid_loader.dataset)
#         valid_losses.append(valid_loss)
#         print(f"Validation Loss: {valid_loss}")
#         early_stopping(valid_loss)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     scheduler.step()

# print("Finished Training")

# plt.plot(train_losses, label="Training loss")
# plt.plot(valid_losses, label="Validation loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# resnet50.eval()
# with torch.no_grad():
#     for images, labels in valid_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = resnet50(images)
#         _, preds = torch.max(outputs, 1)
#         evaluate_model_performance(preds, labels)
#         break

# for image in test_images[:10]:
#     predict_and_visualize(
#         os.path.join(test_dir, "test", image), resnet50, class_names, device
#     )
