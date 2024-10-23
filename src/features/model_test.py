import torch
import timm
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import requests
from io import BytesIO

def get_efficientnetv2_model(output_size: int):
    """
    Instantiates an EfficientNetV2 model with pretrained weights.

    Args:
        output_size (int): Number of output classes.

    Returns:
        nn.Module: EfficientNetV2 model.
    """
    model = timm.create_model(
        "efficientnetv2_rw_s",  # Using EfficientNetV2 RW small version
        pretrained=True,         # Use pretrained weights
        num_classes=output_size, # Adjust output size to match number of classes
    )
    return model

def load_model(model_path, device, num_classes=21):
    """
    Loads the trained EfficientNetV2 model.

    Args:
        model_path (str): Path to the saved model state_dict.
        device (torch.device): Device to load the model on.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Loaded EfficientNetV2 model.
    """
    # Instantiate the model
    model = get_efficientnetv2_model(output_size=num_classes)

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Adjust state_dict if saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove 'module.' prefix
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict

    # Load the adjusted state_dict into the model
    model.load_state_dict(new_state_dict)

    # Move the model to device and set to eval mode
    model.to(device)
    model.eval()

    return model

def load_class_names(labels_mapping_path, num_classes=21):
    """
    Loads class names from the labels mapping JSON.

    Args:
        labels_mapping_path (str): Path to the labels mapping JSON file.
        num_classes (int): Number of classes.

    Returns:
        list: List of class names ordered by index.
    """
    with open(labels_mapping_path, 'r') as f:
        labels_mapping = json.load(f)

    idx_to_disease = {v: k for k, v in labels_mapping['disease_to_idx'].items()}
    class_names = [idx_to_disease[i] for i in range(num_classes)]
    return class_names

def preprocess_image(image, transform):
    """
    Preprocesses the input image.

    Args:
        image (PIL.Image.Image): Input image.
        transform (torchvision.transforms.Compose): Transformations to apply.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    return transform(image).unsqueeze(0)  # Add batch dimension

def download_image(image_url):
    """
    Downloads an image from a given URL.

    Args:
        image_url (str): URL of the image.

    Returns:
        PIL.Image.Image: Downloaded image.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except IOError as e:
        print(f"Error opening image: {e}")
        return None

def run_inference(model, input_batch):
    """
    Runs inference on the input batch.

    Args:
        model (nn.Module): Trained model.
        input_batch (torch.Tensor): Preprocessed input batch.

    Returns:
        tuple: (probabilities, confidence, predicted)
    """
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return probabilities, confidence, predicted

def visualize_prediction(image, predicted_class, confidence_score):
    """
    Visualizes the prediction result.

    Args:
        image (PIL.Image.Image): Input image.
        predicted_class (str): Predicted class name.
        confidence_score (float): Confidence score in percentage.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_class} ({confidence_score:.2f}% confidence)')
    plt.axis('off')
    plt.show()

def visualize_top_k_predictions(image, class_names, top_classes, top_confidences):
    """
    Visualizes the top K predictions.

    Args:
        image (PIL.Image.Image): Input image.
        class_names (list): List of all class names.
        top_classes (list): Top K predicted class names.
        top_confidences (list): Top K confidence scores.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Top Predictions:')
    plt.show()

    for i, (cls, conf) in enumerate(zip(top_classes, top_confidences), 1):
        print(f"{i}. {cls}: {conf:.2f}% confidence")

def main():
    # 1. Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Number of classes
    num_classes = 21

    # 3. Paths
    model_path = "../../models/baseline_models/EfficientNetV2_best_val_loss_model.pth"
    labels_mapping_path = "/home/olarinoyem/Research/plant_disease_dl/data/processed/plant_leaf_disease_dataset/single_task_disease/labels_mapping_single_task_disease.json"

    # 4. Instantiate and load the model
    model = load_model(model_path, device, num_classes=num_classes)

    # 5. Load class names
    class_names = load_class_names(labels_mapping_path, num_classes=num_classes)

    # 6. Define image transformations (ensure these match your training transformations)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Adjust if your model expects a different size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 7. Specify the image source
    # Option A: Download from the internet
    image_url = "https://cropwatch.unl.edu/documents/gray-leaf-spot-lesions-F4.jpg"  # Replace with your image URL

    # Download the image
    image = download_image(image_url)

    if image is not None:
        # 8. Preprocess the image
        input_batch = preprocess_image(image, transform).to(device)

        # 9. Run inference
        probabilities, confidence, predicted = run_inference(model, input_batch)

        # 10. Visualize the results
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100  # Convert to percentage

        visualize_prediction(image, predicted_class, confidence_score)

        # Optional: Top-K predictions visualization
        top_k = 3
        top_probs, top_idxs = torch.topk(probabilities, top_k, dim=1)
        top_classes = [class_names[idx.item()] for idx in top_idxs[0]]
        top_confidences = [prob.item() * 100 for prob in top_probs[0]]
        visualize_top_k_predictions(image, class_names, top_classes, top_confidences)
    else:
        print("Failed to download or open the image. Please check the URL and try again.")

if __name__ == "__main__":
    main()
