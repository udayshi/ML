"""Train and export a TensorFlow image classifier."""

from __future__ import annotations


import json
from pathlib import Path

import tensorflow as tf  # type: ignore[import-untyped]

# Settings
# Image height in pixels.
IMAGE_HEIGHT = 128
# Image width in pixels.
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1  # Grayscale uses one brightness channel instead of RGB's three.
# Number of images processed together.
BATCH_SIZE = 32
# Number of times the model sees the training dataset.
EPOCHS = 10
TRAIN_DIRECTORY = Path("datasets/training_set")
VALIDATION_DIRECTORY = Path("datasets/test_set")
MODEL_PATH = Path("models/cnn-demo.model")
LABELS_PATH = MODEL_PATH / "labels.json"

# Load training images. Every subfolder is automatically a class.
if not TRAIN_DIRECTORY.is_dir():
    # Stop with a clear error if the training folder is missing.
    raise FileNotFoundError(f"Training directory does not exist: {TRAIN_DIRECTORY}")

# Find images and labels from the training subfolders.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    # Folder containing one subfolder for each class.
    TRAIN_DIRECTORY,
    # Resize every image to the same size.
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # Load this many images in one batch.
    batch_size=BATCH_SIZE,
    # Randomize training image order.
    shuffle=True,
    # Decode as RGB first. TensorFlow's BMP decoder does not support channels=1.
    color_mode="rgb",
)
# Read the class names from the discovered subfolders.
class_names = list(train_dataset.class_names)
# Use the number of folders to size the model's output layer.
class_count = len(class_names)

if class_count < 2:
    # A classifier needs at least two possible classes.
    raise ValueError(f"Expected at least two class folders, received {class_count}.")
# Show the classes so the folder setup can be checked.
print(f"Classes found: {class_names}")

# Load validation images and make sure their folders match the training folders.
if not VALIDATION_DIRECTORY.is_dir():
    # Stop with a clear error if the validation folder is missing.
    raise FileNotFoundError(f"Validation directory does not exist: {VALIDATION_DIRECTORY}")

# Load separate images used to check the model during training.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    # Folder containing validation class subfolders.
    VALIDATION_DIRECTORY,
    # Use the same image size as the training data.
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # Use the same batch size as training.
    batch_size=BATCH_SIZE,
    # Validation order does not need to be randomized.
    shuffle=False,
    # Decode as RGB first. TensorFlow's BMP decoder does not support channels=1.
    color_mode="rgb",
)
# Make sure the validation folders have the same order and names as training folders.
if list(validation_dataset.class_names) != class_names:
    # Stop rather than using the wrong label for a prediction.
    raise ValueError("Training and validation directories must contain the same class folders.")

# Convert RGB tensors to grayscale after decoding. This works for JPG, PNG, and BMP files.
# Keep labels unchanged while converting the image tensors.
train_dataset = train_dataset.map(
    lambda images, labels: (tf.image.rgb_to_grayscale(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Apply the same conversion to validation images.
validation_dataset = validation_dataset.map(
    lambda images, labels: (tf.image.rgb_to_grayscale(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prepare future batches while the current batch is being processed.
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# Build the neural network. Images move through these layers from top to bottom.
# The final layer uses class_count, so new folders automatically add outputs.
model = tf.keras.Sequential(
    [
        # Tell the model the shape of one grayscale image.
        tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
        # Change pixel values from 0-255 to 0-1.
        tf.keras.layers.Rescaling(1.0 / 255.0),
        # Detect visual patterns.
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        # Reduce the feature-map size.
        tf.keras.layers.MaxPooling2D(),
        # Detect more complex patterns.
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        # Reduce the feature-map size again.
        tf.keras.layers.MaxPooling2D(),
        # Detect higher-level patterns.
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        # Reduce the feature-map size one more time.
        tf.keras.layers.MaxPooling2D(),
        # Turn feature maps into one list of values.
        tf.keras.layers.Flatten(),
        # Combine the detected features.
        tf.keras.layers.Dense(128, activation="relu"),
        # Produce one probability for every discovered class.
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ]
)

# Configure and train the model.
model.compile(
    # Adam updates the model weights after each batch.
    optimizer="adam",
    # Compare numeric labels with predicted probabilities.
    loss="sparse_categorical_crossentropy",
    # Display the percentage of correct predictions.
    metrics=["accuracy"],
)
# Tell the user that training is starting.
print("Starting training...")
# Train with the training data and check progress using validation data.
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    shuffle=False,  # The training dataset was already shuffled when it was loaded.
)

# Save the trained model and the class names used by its output numbers.
# Create the parent models folder if necessary.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
# Export the model in TensorFlow SavedModel format.
model.export(str(MODEL_PATH))
# Save the class names in the same order as the model's output numbers.
LABELS_PATH.write_text(
    json.dumps({"class_names": class_names}, indent=2),
    encoding="utf-8",
)
# Display the output locations.
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved class names to: {LABELS_PATH}")
