# Plant Disease Classification and Detection

Plant diseases pose a significant threat to global agriculture, leading to substantial losses in crop yield and quality. Early and accurate detection of these diseases is crucial for effective management and prevention. Traditional methods rely on manual inspection by experts, which can be time-consuming, costly, and susceptible to human error. Leveraging advancements in digital image processing and deep learning, this project aims to develop an automated, robust, and accurate plant disease detection system.

This repository encompasses comprehensive code and resources for research on plant disease classification and detection using various deep learning models. The project focuses on building and evaluating models for the following tasks:

- **Crop Type Classification**: Identifying the type of crop from an image.
- **Disease Detection**: Determining whether a plant is healthy or diseased.
- **Disease Type Classification**: Identifying the specific disease affecting the plant.

Enhanced features include:
- **Weights & Biases (W&B) Integration**: For experiment tracking, visualization, and logging.
- **Class Imbalance Handling**: Utilizing `WeightedRandomSampler` to address imbalanced datasets.
- **Mixed Precision Training**: Leveraging PyTorch's Automatic Mixed Precision (AMP) for efficient training.
- **Callbacks for Training Monitoring**: Implementing early stopping and model checkpointing based on validation loss.
- **Comprehensive Visualization**: Saving artifacts such as training metrics, classification reports, and confusion matrices for detailed analysis.

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
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Overview

Effective management of plant diseases is pivotal in ensuring agricultural productivity and food security. This project harnesses the power of deep learning to build models capable of:

- **Classifying Crop Types**: Differentiating between various crop species based on image data.
- **Detecting Plant Health**: Identifying whether a plant is healthy or exhibiting disease symptoms.
- **Classifying Disease Types**: Pinpointing specific diseases affecting the plants.

By automating these tasks, the system aims to provide timely and accurate insights, aiding farmers and agricultural experts in decision-making processes.

## Dataset

The project utilizes the [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset) and the [Plant Village Dataset on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1). These datasets comprise images of healthy and diseased plant leaves across multiple crop types and disease categories.

- **Total Classes**: 38 (including both healthy and diseased states).
- **Plant Types**: Multiple species such as apple, corn, grape, etc.
- **Diseases**: Various diseases like Apple Scab, Grape Black Rot, etc.

**Note**: Due to the size of the datasets, they are not included in this repository. Please download them separately and place them in the designated directories as outlined in the Data Preparation section.

## Getting Started

### Prerequisites

Ensure the following software and libraries are installed on your system:

- **Python**: Version 3.7 or higher
- **PyTorch**: Version 1.7 or higher
- **Torchvision**
- **Timm**: For pre-trained models like EfficientNetV2
- **Weights & Biases (W&B)**: For experiment tracking
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **TQDM**: For progress bars

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   *`requirements.txt` should include all necessary dependencies, for example:*

   ```plaintext
   torch
   torchvision
   timm
   wandb
   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn
   tqdm
   ```

4. **Configure Weights & Biases**

   Sign up for a [Weights & Biases](https://wandb.ai/) account if you haven't already. After installation, log in via the terminal:

   ```bash
   wandb login
   ```

   Enter your W&B API key when prompted.

### Data Preparation

#### Download the Dataset

1. **New Plant Diseases Dataset (Augmented) on Kaggle**

   - **Link**: [Kaggle Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
   - **Download** the dataset and extract it.

2. **Plant Village Dataset on Mendeley Data**

   - **Link**: [Plant Village Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)
   - **Download** the dataset and extract it.

#### Creating Datasets

Scripts are provided to preprocess the raw datasets into formats suitable for training and evaluation.

1. **Single-Task Classification Dataset**

   For tasks like disease type classification.

   ```bash
   python scripts/create_single_task_dataset.py --input_dir data/raw/new-plant-diseases-dataset --output_dir data/processed/single_task_disease --train_split 0.8
   ```

2. **Multi-Task Classification Dataset**

   For simultaneous crop type classification, disease detection, and disease type classification.

   ```bash
   python scripts/create_multitask_dataset.py --input_dir data/raw/new-plant-diseases-dataset --output_dir data/processed/multi_task_classification --train_split 0.8
   ```

   **Arguments**:

   - `--input_dir`: Path to the raw dataset directory.
   - `--output_dir`: Path to save the processed dataset.
   - `--train_split`: Fraction of data to use for training (e.g., 0.8 for 80%).

**Note**: Ensure that the directory structure of the raw datasets aligns with the expectations of the preprocessing scripts.

## Modeling

### Models Implemented

The project explores both baseline convolutional neural network (CNN) models and advanced architectures for plant disease classification and detection.

1. **Baseline Models**

   - **BaselineModel**: A simple CNN with basic convolutional and pooling layers.
   - **ConvNetPlus**: An enhanced CNN with additional layers, batch normalization, and dropout for better generalization.
   - **TinyVGG**: A compact version of the VGG network tailored for the dataset.
   - **EfficientNetV2**: A state-of-the-art model leveraging pre-trained weights for improved performance.

2. **Vision Transformer (ViT)**

   - **VisionTransformer**: Implements the Vision Transformer architecture from scratch, designed to handle image classification tasks by dividing images into patches and processing them through transformer encoder blocks.

### Training the Models

Two separate scripts are provided for training:

1. **Training Vision Transformer (`train_vit_model.py`)**

   Handles the training of the Vision Transformer model with integrated features like W&B tracking, mixed precision training, and callbacks.

   **Usage**:

   ```bash
   python train_vit_model.py
   ```

2. **Training Baseline Models (`train_baseline_models.py`)**

   Facilitates the training of all baseline models, each with its specific configurations and optimizations.

   **Usage**:

   ```bash
   python train_baseline_models.py
   ```

**Features**:

- **Weights & Biases Integration**: Tracks experiments, logs metrics, and saves artifacts for comprehensive analysis.
- **Class Imbalance Handling**: Utilizes `WeightedRandomSampler` to ensure balanced representation of classes during training.
- **Mixed Precision Training**: Enhances training efficiency and reduces memory consumption without sacrificing performance.
- **Callbacks for Training Monitoring**: Implements early stopping to prevent overfitting and model checkpointing to retain the best-performing models.

### Evaluation

Post-training evaluation scripts generate detailed metrics and visualizations to assess model performance.

1. **Evaluation Metrics**

   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**
   - **Confusion Matrix**

2. **Running Evaluation**

   The evaluation is integrated within the training scripts and logs results to W&B. Additionally, confusion matrices and classification reports are saved locally for further inspection.

   **Example Output**:

   ```bash
   # After training, evaluation metrics are automatically generated and logged.
   ```

## Results

Our experiments showcase the performance of both baseline models and the Vision Transformer on the tasks of crop type classification, disease detection, and disease type classification.

- **Baseline Models**:
  - **BaselineModel**: Achieved reasonable accuracy with a simple architecture.
  - **ConvNetPlus**: Demonstrated improved performance due to deeper architecture and regularization techniques.
  - **TinyVGG**: Balanced model complexity and performance effectively.
  - **EfficientNetV2**: Leveraged pre-trained weights to attain superior accuracy and generalization.

- **Vision Transformer (ViT)**:
  - Exhibited strong performance by effectively capturing global image features through transformer encoder blocks.
  - Benefited from class-specific augmentations and mixed precision training for enhanced efficiency and accuracy.

Detailed results, including training curves, accuracy metrics, and confusion matrices, are available in the `reports/` directory and visualized through Weights & Biases dashboards.

## Visualization

Comprehensive visualization tools are integrated to facilitate in-depth analysis of model performance.

1. **Training and Validation Metrics**

   - **Loss and Accuracy Curves**: Plotting training and validation loss and accuracy over epochs to monitor model convergence and detect overfitting.

2. **Confusion Matrices**

   - Visual representations of true vs. predicted classifications across all classes, aiding in identifying model strengths and weaknesses.

3. **Sample Predictions**

   - Displaying example images alongside true labels and model predictions to qualitatively assess performance.

**Accessing Visualizations**:

- **Local Artifacts**: Saved in the `reports/figures/` directory.
- **Weights & Biases Dashboards**: Accessible through your W&B account, providing interactive and real-time visualizations.

**Example Visualization Script**:

```bash
python src/visualization/plot_results.py --model_name BaselineModel
```

## Project Structure

```plaintext
├── data
│   ├── processed
│   │   ├── single_task_disease
│   │   └── multi_task_classification
│   └── raw
│       ├── new-plant-diseases-dataset
│       └── plant_village_dataset
├── models
│   └── baseline_models
│       ├── BaselineModel_best_val_loss_model.pth
│       ├── ConvNetPlus_best_val_loss_model.pth
│       ├── TinyVGG_best_val_loss_model.pth
│       └── EfficientNetV2_best_val_loss_model.pth
├── notebooks
│   └── exploratory_analysis.ipynb
├── reports
│   ├── figures
│   │   ├── BaselineModel_training_validation_metrics.png
│   │   ├── ConvNetPlus_training_validation_metrics.png
│   │   ├── TinyVGG_training_validation_metrics.png
│   │   ├── EfficientNetV2_training_validation_metrics.png
│   │   └── ViT_training_validation_metrics.png
│   └── results
│       └── model_performance_summary.md
├── src
│   ├── data
│   │   ├── create_single_task_dataset.py
│   │   └── create_multitask_dataset.py
│   ├── models
│   │   ├── BaselineModel.py
│   │   ├── ConvNetPlus.py
│   │   ├── TinyVGG.py
│   │   ├── EfficientNetV2.py
│   │   ├── VisionTransformer.py
│   │   ├── train_vit_model.py
│   │   ├── train_baseline_models.py
│   │   └── model_definitions.py
│   ├── visualization
│   │   └── plot_results.py
│   └── helper_functions.py
├── scripts
│   ├── create_single_task_dataset.py
│   └── create_multitask_dataset.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Contributing

Contributions are highly welcome! If you have suggestions, improvements, or feature requests, please feel free to submit a pull request or open an issue.

**Steps to Contribute**:

1. **Fork the Repository**

   Click on the "Fork" button at the top right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/your_username/plant-disease-classification.git
   cd plant-disease-classification
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes**

   Implement your feature or fix.

5. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request**

   Navigate to your fork on GitHub and click the "Compare & pull request" button.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **Datasets**:
  - [New Plant Diseases Dataset (Augmented) on Kaggle](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
  - [Plant Village Dataset on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

- **Libraries and Frameworks**:
  - [PyTorch](https://pytorch.org/)
  - [Timm](https://github.com/rwightman/pytorch-image-models)
  - [Weights & Biases](https://wandb.ai/)

- **Inspiration**:
  - Contributions and insights from the deep learning community and various online resources.

- **Contact**:
  - For any questions or inquiries, please contact [ajaoolarinoyemichael@gmail.com](mailto:ajaoolarinoyemichael@gmail.com).

Repository: [https://github.com/michaelajao/plant_disease_dl](https://github.com/michaelajao/plant_disease_dl)
```
