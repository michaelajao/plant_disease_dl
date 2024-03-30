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
from torchvision.utils import make_grid
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
plt.savefig("../../reports/figures/disease_count.png")
plt.show()

print(
    f"Total number of images available for training: {disease_count.no_of_images.sum()}"
)

# Get 5 random healthy plant images
healthy_plants = data[data["disease"] == "healthy"].sample(5)

# Get 5 random diseased plant images
diseased_plants = data[data["disease"] != "healthy"].sample(5)

# Plot healthy plant images
plt.figure(figsize=(20, 18))
for i, row in enumerate(healthy_plants.iterrows()):
    plant = row[1]["plant"]
    disease = row[1]["disease"]
    image_path = os.path.join(train_dir, f"{plant}___{disease}")
    image_file = random.choice(os.listdir(image_path))
    image = Image.open(os.path.join(image_path, image_file))
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(f"Healthy {plant}\nDisease: {disease}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("../../reports/figures/healthy_plants.png")     
plt.show()

# Plot diseased plant images
plt.figure(figsize=(20, 18))
for i, row in enumerate(diseased_plants.iterrows()):
    plant = row[1]["plant"]
    disease = row[1]["disease"]
    image_path = os.path.join(train_dir, f"{plant}___{disease}")
    image_file = random.choice(os.listdir(image_path))
    image = Image.open(os.path.join(image_path, image_file))
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(f"Diseased {plant}\nDisease: {disease}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("../../reports/figures/diseased_plants.png")
plt.show()