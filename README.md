# Plant Disease Classification and Detection
Plant diseases are a major threat to crop yield and quality, and their early detection is essential for effective management and prevention. However, traditional methods of plant disease detection rely on manual inspection by experts, which is costly, time-consuming, and prone to errors. Therefore, there is a need for an automated, robust, and accurate plant disease detection system that can leverage the advances in digital image processing and deep learning.

This repository contains the code and resources for our research on plant disease classification and detection using deep learning models. The project focuses on building models for:

- **Crop Type Classification**: Identifying the type of crop from an image.
- **Disease Detection**: Determining whether a plant is healthy or diseased.
- **Disease Type Classification**: Identifying the specific disease affecting the plant.

The repository includes scripts for data preprocessing, model training, evaluation, and visualization of results.

## Table of Contents

- [Plant Disease Classification and Detection](#plant-disease-classification-and-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
      - [Download the Dataset](#download-the-dataset)
      - [Creating Datasets](#creating-datasets)
  - [Modeling](#modeling)
    - [Models Implemented](#models-implemented)
    - [Training the Models](#training-the-models)
    - [Evaluation](#evaluation)
  - [Results](#results)
  - [Visualization](#visualization)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Overview

Plant diseases can cause significant losses in agricultural production. Early detection and accurate diagnosis are crucial for effective management and control of plant diseases. This project aims to leverage deep learning techniques to build robust models capable of:

- Classifying crop types from images.
- Detecting whether a plant is healthy or diseased.
- Identifying the specific disease affecting the plant.

The models are trained and evaluated on a comprehensive dataset containing images of various plants with healthy and diseased samples.

## Dataset

The dataset used in this project is the New Plant Diseases Dataset (Augmented), which contains images of healthy and diseased plant leaves categorized into folders by crop type and disease name.

- **Total Classes**: 38 different classes (including healthy and diseased states).
- **Plant Types**: Multiple plant species such as apple, corn, grape, etc.
- **Diseases**: Various diseases affecting the plants, e.g., Apple Scab, Grape Black Rot.

**Note**: Due to the size of the dataset, it is not included in this repository. Please download it separately and place it in the appropriate directory as instructed in the Data Preparation section.

dataset: [New Plant Diseases Dataset (Augmented) on Kaggle](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
dataset: [Plant village dataset on mendeley data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.7 or higher
- PyTorch 1.7 or higher
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- seaborn
- tqdm

### Installation

Clone the Repository:

```bash
git clone https://github.com/your_username/plant-disease-classification.git
cd plant-disease-classification
```

Create a Virtual Environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Required Packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

#### Download the Dataset

Download the Dataset:

- [New Plant Diseases Dataset (Augmented) on Kaggle](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

Extract the Dataset:

Unzip the dataset and place it in the `data/raw/` directory.

```bash
# Assuming you are in the root directory of the project
unzip new-plant-diseases-dataset.zip -d data/raw/
```

#### Creating Datasets

We have provided scripts to create datasets for both normal classification and multi-task classification problems.

1. **Normal Classification Dataset**

Creates a dataset suitable for single-task classification (e.g., disease classification).

```bash
python scripts/create_normal_classification_dataset.py --num_classes 10 --train_split 0.8
```

- `--num_classes`: Number of classes to include (default: all classes).
- `--train_split`: Fraction of data to use for training (default: 0.8).

The processed dataset will be saved in `data/processed/normal_classification/`.

2. **Multi-Task Classification Dataset**

Creates a dataset suitable for multi-task classification, handling crop type classification, disease detection, and disease type classification.

```bash
python scripts/create_multitask_classification_dataset.py --num_plants 5 --num_diseases 5 --train_split 0.8
```

- `--num_plants`: Number of plant types to include (default: all plants).
- `--num_diseases`: Number of diseases to include (default: all diseases).
- `--train_split`: Fraction of data to use for training (default: 0.8).

The processed dataset will be saved in `data/processed/multitask_classification/`.

## Modeling

### Models Implemented

We have implemented several models for experimentation:

- **BaselineModel**: A simple CNN with basic convolutional and pooling layers.
- **ConvNetPlus**: An improved CNN with additional layers, batch normalization, and dropout.
- **TinyVGG**: A smaller version of the VGG network adapted for our dataset.
- **MultiTaskCNN**: A model designed for multi-task learning, utilizing a shared feature extractor with task-specific heads.

### Training the Models

Scripts for training the models are provided in the `src/models/` directory.

**Training Example**

```bash
python src/models/train_model.py --model_name BaselineModel --epochs 10
```

- `--model_name`: Name of the model to train (BaselineModel, ConvNetPlus, TinyVGG, MultiTaskCNN).
- `--epochs`: Number of epochs to train.

Training logs, models, and figures will be saved in the `models/` and `reports/` directories.

### Evaluation

After training, evaluate the models using the provided evaluation scripts.

```bash
python src/models/evaluate_model.py --model_name BaselineModel
```

- Generates metrics like accuracy, precision, recall, and F1-score.
- Saves confusion matrices and other visualizations.

## Results

Our experiments show the performance of different models on the tasks of crop type classification, disease detection, and disease classification.

- **BaselineModel**:
  - Achieved reasonable accuracy but limited by its simple architecture.
- **ConvNetPlus**:
  - Improved performance due to deeper architecture and regularization techniques.
- **TinyVGG**:
  - Provided a good balance between model complexity and performance.
- **MultiTaskCNN**:
  - Effectively handled multiple tasks simultaneously, leveraging shared features.

Detailed results, including training curves and confusion matrices, are available in the `reports/` directory.

## Visualization

We provide visualization scripts to generate plots and figures for analysis.

- **Training and Validation Loss and Accuracy**:
  - Plots showing the loss and accuracy over epochs for each model.
- **Confusion Matrices**:
  - Visual representations of model performance across different classes.
- **Sample Predictions**:
  - Images with true labels, predicted labels, and confidence scores.

**Example**:

```bash
python src/visualization/plot_results.py --model_name BaselineModel
```

Figures are saved in the `reports/figures/` directory.

## Usage

To use the trained models for inference or integrate them into your application:

**Load the Model**:

```python
import torch
from src.models.model_definitions import BaselineModel

model = BaselineModel(...)
model.load_state_dict(torch.load('models/BaselineModel_best_model.pth'))
model.eval()
```

**Make Predictions**:

```python
# Assuming 'image' is a preprocessed image tensor
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)
```

## Project Structure

```plaintext
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
├── reports
│   ├── figures
│   └── results
├── src
│   ├── data
│   ├── models
│   ├── visualization
│   └── helper_functions.py
├── scripts
│   ├── create_normal_classification_dataset.py
│   └── create_multitask_classification_dataset.py
├── requirements.txt
└── README.md
```

- `data/`: Contains raw and processed data.
- `models/`: Saved models and training logs.
- `reports/`: Generated analysis and figures.
- `src/`: Source code for data processing, modeling, and visualization.
- `scripts/`: Scripts for dataset creation.
- `notebooks/`: Jupyter notebooks for exploration.
- `requirements.txt`: List of required packages.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The dataset was sourced from Kaggle.
- Inspiration and guidance from the deep learning community and various online resources.
- Contact: For any questions or inquiries, please contact ajaoolarinoyemichael@gmail.com.

Repository: [https://github.com/michaelajao/plant_disease_dl](https://github.com/michaelajao/plant_disease_dl)