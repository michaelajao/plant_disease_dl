# This script is used to visualize the data in the dataset
import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Ensure the helper functions and settings are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helper_functions import *  # Import helper functions and matplotlib settings

# Specify the path to the dataset
data_path = "../../data/raw/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/"
train_dir = os.path.join(data_path, "train")
valid_dir = os.path.join(data_path, "valid")

# Extract disease categories and prepare plant/disease metadata
diseases = os.listdir(train_dir)
plant_names = list(set(disease.split("___")[0] for disease in diseases))
disease_names = list(set(disease.split("___")[1] for disease in diseases))

# Display unique plants and diseases count
print(f"Number of unique plants: {len(plant_names)}")
print(f"Number of unique diseases: {len(disease_names)}")

# Count the number of images for each disease
disease_count = {}
for disease in diseases:
    disease_path = os.path.join(train_dir, disease)
    try:
        num_images = len(os.listdir(disease_path))
        disease_count[disease] = num_images
    except FileNotFoundError:
        print(f"Warning: Directory for {disease} not found.")
        disease_count[disease] = 0  # Set to 0 if directory doesn't exist

# Convert the disease count into a DataFrame for easier plotting
disease_count_df = pd.DataFrame.from_dict(disease_count, orient='index', columns=["no_of_images"])
disease_count_df = disease_count_df.sort_values(by="no_of_images", ascending=False)

# Function to plot bar charts
def plot_bar(df, title, xlabel, ylabel, color, filename, figsize=(12, 15)):
    plt.figure(figsize=figsize)
    df.plot(kind="barh", color=color, legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Plot total disease count
plot_bar(disease_count_df, "Number of Images for Each Disease", "Number of images", "Disease",
         color="skyblue", filename="../../reports/figures/disease_count_horizontal.pdf")

# Separate healthy and unhealthy diseases
healthy_df = disease_count_df[disease_count_df.index.str.contains("healthy", case=False)]
unhealthy_df = disease_count_df[~disease_count_df.index.str.contains("healthy", case=False)]

# Plot healthy and unhealthy diseases
plot_bar(healthy_df, "Number of Images for Healthy Plants", "Number of images", "Plant",
         color="green", filename="../../reports/figures/healthy_disease_count.pdf", figsize=(10, 6))
plot_bar(unhealthy_df, "Number of Images for Unhealthy Plants", "Number of images", "Disease",
         color="red", filename="../../reports/figures/unhealthy_disease_count.pdf", figsize=(10, 12))

# Calculate total number of healthy and unhealthy images and create a pie chart
def plot_pie(labels, sizes, colors, title, filename):
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(title, fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

healthy_total = healthy_df['no_of_images'].sum()
unhealthy_total = unhealthy_df['no_of_images'].sum()
plot_pie(['Healthy', 'Unhealthy'], [healthy_total, unhealthy_total], ['green', 'red'],
         "Proportion of Healthy vs Unhealthy Images", "../../reports/figures/healthy_unhealthy_proportion_pie.pdf")

# Create heatmap for disease counts per plant
disease_count_df['plant'] = disease_count_df.index.str.split("___").str[0]
plant_disease_matrix = disease_count_df.pivot_table(values='no_of_images', index='plant', aggfunc='sum')

def plot_heatmap(matrix, title, filename, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap="YlGnBu", annot=True, fmt="d", cbar=True)
    plt.title(title, fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

plot_heatmap(plant_disease_matrix, "Heatmap of Disease Counts per Plant", "../../reports/figures/disease_heatmap.pdf")

# Define image transformations and load datasets
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

# Print the number of classes
print(f"Number of classes: {len(train_data.classes)}")

# Display a sample image from the dataset
def display_sample_image(image, label, filename):
    plt.imshow(image.permute(1, 2, 0))
    plt.title(train_data.classes[label])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

image, label = train_data[200]
display_sample_image(image, label, "../../reports/figures/sample_image.pdf")

# Display grid of random images
def plot_image_grid(data, nrows=3, ncols=3, filename=None):
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, nrows * ncols + 1):
        image, label = data[np.random.randint(len(data))]
        ax = fig.add_subplot(nrows, ncols, i)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

plot_image_grid(train_data, nrows=3, ncols=3, filename="../../reports/figures/random_image_grid.pdf")

# Load data into DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

# Print data loader details
print(f"Length of train_loader: {len(train_loader)} with batch size of {BATCH_SIZE}")
print(f"Length of valid_loader: {len(valid_loader)} with batch size of {BATCH_SIZE}")

# Display 5 random images with labels from a batch
def show_batch_images(images, labels, ncols=5):
    random_indices = np.random.choice(len(images), size=5, replace=False)
    fig, axes = plt.subplots(1, ncols, figsize=(15, 3))
    for i, ax in enumerate(axes):
        idx = random_indices[i]
        image = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(f"Label: {labels[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

images, labels = next(iter(train_loader))
show_batch_images(images, labels)
