"""Load the PyTorch image classifier and classify one image."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Settings
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1
MODEL_PATH = Path("models/cnn-demo-torch.pt")
IMAGE_PATH = Path("datasets/single_prediction/cat/10.jpg")
LABELS_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.labels.json")

# Check the files needed for prediction.
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image does not exist: {IMAGE_PATH}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Saved model does not exist: {MODEL_PATH}")
if not LABELS_PATH.is_file():
    raise FileNotFoundError(f"Labels file does not exist: {LABELS_PATH}")

# Read the class names saved by cnn_build_torch.py.
labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
class_names = labels_data.get("class_names")
if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
    raise ValueError(f"Invalid class names in {LABELS_PATH}")

# Build the same model used during training.
model = torch.nn.Sequential(
    torch.nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(32, 64, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(64, 128, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(128 * 14 * 14, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, len(class_names)),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# Load and resize the image to the same size used during training.
image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
        transforms.ToTensor(),
    ]
)

for i in range(10):
    id=i+1
    IMAGE_PATH = Path(f"datasets/single_prediction/dog/{id}.jpg")

    image = Image.open(IMAGE_PATH)
    image_batch = image_transform(image).unsqueeze(0)

    # Ask the model for one probability per class.
    with torch.no_grad():
        probabilities = torch.softmax(model(image_batch), dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())
    if predicted_index >= len(class_names):
        raise ValueError("The model output does not match the saved class names.")

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())
    print(f"Prediction: {predicted_class} ({confidence:.2%} confidence) id:{id}")
