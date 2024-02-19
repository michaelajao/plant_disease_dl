import pandas as pd
import numpy as np
import os
from PIL import Image
from zipfile import ZipFile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import precision_score, recall_score, f1_score#
from tqdm import tqdm
from torchsummary import summary

# Set the random seed for reproducibility
torch.manual_seed(10)



# Set the style of the plots
plt.style.use("seaborn-v0_8-white")
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


# with ZipFile('../../data/raw/archive.zip', 'r') as zip_ref:
#     zip_ref.extractall('../../data/raw')

print(os.listdir('../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'))


print(len(os.listdir('../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')))

data_path = '../../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'

train_dir = os.path.join(data_path, 'train')
valid_dir = os.path.join(data_path, 'valid')

diseases = os.listdir(train_dir)

# extract the unique plant names 
plant_names = []
for disease in diseases:
    plant = disease.split('___')[0]
    if plant not in plant_names:
        plant_names.append(plant)


# extract the unique disease names
disease_names = []
for disease in diseases:
    disease = disease.split('___')[1]
    if disease not in disease_names:
        disease_names.append(disease)
        
print(f'Number of unique plants: {len(plant_names)}')
print(f'Number of unique diseases: {len(disease_names)}')


# check the number of images for each disease
disease_count = {}
for disease in diseases:
    if disease not in disease_count:
        disease_count[disease] = len(os.listdir(os.path.join(train_dir, disease)))
        
# convert to dataframe
disease_count = pd.DataFrame(disease_count.values(), index=disease_count.keys(), columns=['no_of_images'])


# plot the number of images for each disease

disease_count.sort_values(by='no_of_images', ascending=False).plot(kind='bar', figsize=(15, 8), color='skyblue')
plt.title('Number of images for each disease')
plt.xlabel('Disease')
plt.ylabel('Number of images')
plt.show()


# number of images available for training
print(f'Total number of images available for training: {disease_count.no_of_images.sum()}')

# Dataset preparation for training
# create a dataset object
train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_data = datasets.ImageFolder(valid_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
# check the number of classes
print(f'Number of classes: {len(train_data.classes)}')

#check the shape and vislualize one of the image in the training dataset with the label in text
image, label = train_data[200]
print(f'Image shape: {image.shape}')
print(f'Label: {label}')    
# visualize the image   
plt.imshow(image.permute(1, 2, 0))
plt.title(train_data.classes[label])
plt.show()


# training_sample = 5000
# train_data, _ = random_split(train_data, [training_sample, len(train_data) - training_sample])
# create a dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)


# create helper function to show a batch of training instances
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break
    
show_batch(train_loader)


# Modelling - lets start with using 1000 images from the training dataset to train a simple CNN model using the GPU
# check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device for training.')
# create a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Assuming the input size is 224x224, adjust the linear layer size accordingly
        self.fc1 = nn.Linear(64 * 14 * 14, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


# create a model instance
model = SimpleCNN(in_channels=3, num_classes=len(train_data.classes)).to(device)
print(model)

# visualize the model architecture
print(summary(model, (3, 224, 224)))
# model = SimpleCNN(in_channels=3, num_classes=len(train_data.dataset.classes)).to(device)
# print(model)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

#create a learning rate scheduler


optimizer = optim.Adam(model.parameters(), lr=0.001)
sceduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Implement early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 2
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
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

n_epochs = 50
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

train_losses = []
valid_losses = []

# train the model
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    model.train()
    train_loss = 0.0
    valid_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f'Training Loss: {train_loss}')
    
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print(f'Validation Loss: {valid_loss}')
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    sceduler.step()
    
print('Finished Training')

# plot the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# create a helper function to evaluate the model performance

def evaluate_model_performance(preds, labels):
    """
    Evaluates and prints the model's performance metrics including accuracy,
    precision, recall, and F1 score.
    
    Parameters:
    - preds: The predicted labels (as a PyTorch tensor).
    - labels: The true labels (as a PyTorch tensor).
    """
    # Ensure both predictions and labels are on the CPU
    preds = preds.cpu()
    labels = labels.cpu()

    # Calculate accuracy
    accuracy = torch.sum(preds == labels).item() / len(labels)

    # Convert tensors to numpy arrays for compatibility with scikit-learn
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    # Calculate precision, recall, and F1 score
    precision = precision_score(labels_np, preds_np, average='macro')
    recall = recall_score(labels_np, preds_np, average='macro')
    f1 = f1_score(labels_np, preds_np, average='macro')

    # Print metrics
    print(f'Predicted labels: {preds}')
    print(f'Actual labels: {labels}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


# evaluate the model on the validation dataset
model.eval()
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        evaluate_model_performance(preds, labels)
        break
    
    
test_dir = "../../data/raw/test/"
test = datasets.ImageFolder(test_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

test_images = os.listdir(os.path.join(test_dir, 'test'))
print(f'Number of test images: {len(test_images)}')

# predict the labels for 10 of the test images with the accuracy of prediction. create a function to visualize the predictions
def predict_and_visualize(image_path, model, class_names, device):
    """
    Predict and visualize an image along with its predicted class and accuracy percentage.

    Parameters:
    - image_path: Path to the image file.
    - model: Trained deep learning model.
    - class_names: List of class names corresponding to model outputs.
    - device: The torch device to use for computations.
    """
    # Transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, dim=1)
    
    # Visualize
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[top_catid.item()]} - {top_prob.item()*100:.2f}%")
    plt.axis('off')
    plt.show()


class_names = train_data.classes
for image in test_images[:10]:
    predict_and_visualize(os.path.join(test_dir, 'test', image), model, class_names, device)          
    
    
# save the model
torch.save(model.state_dict(), '../../models/plant_disease_model.pth')  



import torchvision.models as models

# # Load a pre-trained ResNet-50 model
# resnet50 = models.resnet50(pretrained=True)

# # To use the model for inference
# resnet50.eval()

# transfer learning with resnet50

# Load the pre-trained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Freeze the model parameters
for param in resnet50.parameters():
    param.requires_grad = False
    
# Replace the final fully connected layer
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, len(train_data.classes))

# Move the model to the GPU
resnet50 = resnet50.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
sceduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
n_epochs = 50
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

train_losses = []
valid_losses = []

# Train the model
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    resnet50.train()
    train_loss = 0.0
    valid_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet50(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f'Training Loss: {train_loss}')
    
    resnet50.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print(f'Validation Loss: {valid_loss}')
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    sceduler.step()
    
print('Finished Training')

# Plot the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation dataset
resnet50.eval()
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50(images)
        _, preds = torch.max(outputs, 1)
        evaluate_model_performance(preds, labels)
        break
    
# predict the labels for 10 of the test images with the accuracy of prediction. create a function to visualize the predictions

for image in test_images[:10]:
    predict_and_visualize(os.path.join(test_dir, 'test', image), resnet50, class_names, device)


    
    