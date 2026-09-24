"""Train and save the PyTorch image classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Settings
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 200
TRAIN_DIRECTORY = Path("datasets/training_set")
VALIDATION_DIRECTORY = Path("datasets/test_set")
MODEL_PATH = Path("models/cnn-demo-torch.pt")
LABELS_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.labels.json")

# Load training images. Every subfolder is automatically a class.
if not TRAIN_DIRECTORY.is_dir():
    raise FileNotFoundError(f"Training directory does not exist: {TRAIN_DIRECTORY}")

image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(TRAIN_DIRECTORY, transform=image_transform)
class_names = train_dataset.classes
class_count = len(class_names)

if class_count < 2:
    raise ValueError(f"Expected at least two class folders, received {class_count}.")
print(f"Classes found ({class_count}): {class_names}")
print(f"Class mapping: {train_dataset.class_to_idx}")

# Load validation images and make sure their folders match the training folders.
if not VALIDATION_DIRECTORY.is_dir():
    raise FileNotFoundError(f"Validation directory does not exist: {VALIDATION_DIRECTORY}")

validation_dataset = datasets.ImageFolder(
    VALIDATION_DIRECTORY,
    transform=image_transform,
)
if validation_dataset.classes != class_names:
    raise ValueError("Training and validation directories must contain the same class folders.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches per epoch: {len(validation_loader)}")
print(f"Image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# Build the neural network. Images move through these layers from top to bottom.
model = nn.Sequential(
    nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(128 * 14 * 14, 128),
    nn.ReLU(),
    nn.Linear(128, class_count),
)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print("Model architecture:")
print(model)
print(f"Trainable parameters: {parameter_count:,}")

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Configure and train the model.
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    training_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        prediction = model(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    model.eval()
    validation_correct = 0
    validation_total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            prediction = model(images)
            validation_correct += int((prediction.argmax(dim=1) == labels).sum().item())
            validation_total += labels.size(0)

    validation_accuracy = validation_correct / validation_total
    average_training_loss = training_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"training loss: {average_training_loss:.4f}, "
        f"validation accuracy: {validation_accuracy:.2%}"
    )
    if validation_accuracy>=80:
        break

# Save the trained model and the class names used by its output numbers.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
LABELS_PATH.write_text(
    json.dumps({"class_names": class_names}, indent=2),
    encoding="utf-8",
)
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved class names to: {LABELS_PATH}")
