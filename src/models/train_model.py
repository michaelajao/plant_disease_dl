# Importing necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from timeit import default_timer as timer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# load in the helper functions
from helper_functions import set_seeds, accuracy_fn
from helper_functions import *
# Set the seed for reproducibility
set_seeds(42)

# setup hyperparameters and configurations
BATCH_SIZE = 32
LEARNING_RATE = 0.1
NUM_EPOCHS = 5
Height = 224
Width = 224

# Check if CUDA is available and configure accordingly
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")  # Use CUDA if available
else:
    device = torch.device("cpu")  # Otherwise, use CPU

# Specify the path to the dataset
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Extract disease categories
diseases = os.listdir(train_dir)

# Get unique plant names
plant_names = list(set(disease.split("___")[0] for disease in diseases))

# Get unique disease names
disease_names = list(set(disease.split("___")[1] for disease in diseases))

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((Height, Width)),
        transforms.ToTensor(),
        # Uncomment the following lines for data augmentation during training
        # transforms.RandomRotation(degrees=45),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the training and validation datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

# Print the number of classes in the dataset
print(f"Number of classes: {len(train_data.classes)}")

# Visualize a sample image from the training dataset
image, label = train_data[200]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")




train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

# print the length of the train and valid loaders and the batchs of the train loader
print(f"length of train_loader: {len(train_loader)} with batch size of {BATCH_SIZE}")
print(f"length of valid_loader: {len(valid_loader)} with batch size of {BATCH_SIZE}")

for images, labels in train_loader:
    print(f"Image batch dimensions: {images.shape}")
    print(f"Label batch dimensions: {labels.shape}")
    break



# BaseLine Model 
# create a flatten layer to flatten the input image
flatten_model = nn.Flatten()

# get a single batch of images from the train_loader
x = train_data[0][0].unsqueeze(0)  # Add an extra dimension to simulate a batch of 1 image

# Flatten the image
x_flattened = flatten_model(x)

# Print the shape of the original and flattened images
print(f"Original image shape: {x.shape}")
print(f"Flattened image shape: {x_flattened.shape}")

# create a base line model with two linear layers
class BaselineModel(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(BaselineModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        
    def forward(self, x):
        x = self.layer_stack(x)
        return x
    
    
# Model instantiation
input_size = 3 * Height * Width
hidden_size = 10
output_size = len(train_data.classes)

model_0 = BaselineModel(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_0.parameters(), lr=LEARNING_RATE)

# def print_train_time(start, end, device):
#     train_time = end - start
#     print(f"Training time: {train_time:.4f} seconds on {device}")
#     return train_time
# # Training loop
# start_time = timer()


# end_time = timer()
# print_train_time(start_time, end_time, device="GPU")

# create the training loop and validation loop
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    model_0.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_0(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100 * train_correct / train_total
    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    model_0.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_0(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100 * valid_correct / valid_total
    print(f"Epoch {epoch+1}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")






# # Model definition
# class CustomCNN(nn.Module):
#     def __init__(self, num_layers, hidden_units, num_classes):
#         super(CustomCNN, self).__init__()
#         self.features = self._make_layers(num_layers, hidden_units)
#         in_features = hidden_units[-1] * (224 // 2**num_layers) ** 2
#         self.classifier = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     def _make_layers(self, num_layers, hidden_units):
#         layers = []
#         in_channels = 3
#         for i in range(num_layers):
#             layers += [
#                 nn.Conv2d(in_channels, hidden_units[i], kernel_size=3, padding=1),
#                 nn.BatchNorm2d(hidden_units[i]),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#             ]
#             in_channels = hidden_units[i]
#         return nn.Sequential(*layers)


# # Model instantiation
# num_classes = len(train_data.classes)
# model = CustomCNN(
#     num_layers=5, hidden_units=[32, 64, 128, 256, 512], num_classes=num_classes
# ).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# from torch.utils.tensorboard import SummaryWriter

# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter("runs/fashion_mnist_experiment_1")


# # Early stopping
# class EarlyStopping:
#     def __init__(self, patience=1, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0


# # Training loop
# def train(
#     model, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs, device
# ):
#     early_stopping = EarlyStopping(patience=3, min_delta=0.0)
#     train_losses = [[], []]
#     train_accuracies = [[], []]
#     for epoch in range(n_epochs):
#         print(f"Epoch {epoch + 1}\n-------------------------------")
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()

#         train_loss /= len(train_loader.dataset)
#         train_losses[0].append(train_loss)
#         train_accuracy = 100 * train_correct / train_total
#         train_accuracies[0].append(train_accuracy)

#         model.eval()
#         valid_loss = 0.0
#         valid_correct = 0
#         valid_total = 0
#         with torch.no_grad():
#             for images, labels in valid_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 valid_loss += loss.item() * images.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 valid_total += labels.size(0)
#                 valid_correct += (predicted == labels).sum().item()

#         valid_loss /= len(valid_loader.dataset)
#         train_losses[1].append(valid_loss)
#         valid_accuracy = 100 * valid_correct / valid_total
#         train_accuracies[1].append(valid_accuracy)

#         print(
#             f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}"
#         )
#         print(
#             f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {valid_accuracy:.2f}%"
#         )

#         early_stopping(valid_loss)
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break

#         scheduler.step()

#     return model, train_losses, train_accuracies


# model, losses, accuracies = train(
#     model,
#     train_loader,
#     valid_loader,
#     criterion,
#     optimizer,
#     scheduler,
#     n_epochs=20,
#     device=device,
# )

# # Plotting training and validation losses and accuracies
# plt.figure(figsize=(15, 10))
# plt.subplot(1, 2, 1)
# plt.plot(losses[0], label="Training loss")
# plt.plot(losses[1], label="Validation loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(accuracies[0], label="Training accuracy")
# plt.plot(accuracies[1], label="Validation accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("../../reports/figures/model_training_performance.pdf")
# plt.show()


# def evaluate_model_performance(model, data_loader, device):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for images, labels in data_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.view(-1).cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average="macro")
#     recall = recall_score(all_labels, all_preds, average="macro")
#     f1 = f1_score(all_labels, all_preds, average="macro")
#     print(
#         f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
#     )


# evaluate_model_performance(model, valid_loader, device)


# # Visualizing predictions make it a plot side by side with the probability of the prediction. make the plot good enough to be saved as an image for my research paper
# path_test = "../../data/raw/"

# # test images transformation
# # transform = transforms.Compose(
# #     [
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         # transforms.Normalize(
# #         #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# #         # ),
# #     ]
# # )

# test_data = datasets.ImageFolder(path_test, transform=transform)

# test_loader = DataLoader(test_data, batch_size=32)


# def show_predictions(model, data_loader, device, class_names):
#     model.eval()
#     with torch.no_grad():
#         for images, labels in data_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             probs = F.softmax(outputs, dim=1)
#             fig, axes = plt.subplots(2, 5, figsize=(25, 10))
#             for i, ax in enumerate(axes.flat):
#                 ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
#                 ax.axis("off")
#                 pred_label = class_names[preds[i]]
#                 pred_prob = probs[i][preds[i]] * 100
#                 ax.set_title(f"Prediction: {pred_label}\nProbability: {pred_prob:.2f}%")
#             plt.tight_layout()
#             plt.savefig("../../reports/figures/custom_model_predictions.pdf")
#             plt.show()
#             break


# show_predictions(model, test_loader, device, train_data.classes)


# # Save the model
# torch.save(model.state_dict(), "../../models/plant_disease_model.pth")

# # load the saved trained model

# model = torch.load("../../models/plant_disease_model.pth")
