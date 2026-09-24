"""Load a saved TensorFlow model and classify one image."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

# Settings
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
MODEL_PATH = Path("models/cnn-demo.model")
IMAGE_PATH = Path("datasets/single_prediction/d-1.jpg")
LABELS_PATH = MODEL_PATH / "labels.json"

# Check the files needed for prediction.
if not IMAGE_PATH.is_file():
    # Stop if the selected image cannot be found.
    raise FileNotFoundError(f"Image does not exist: {IMAGE_PATH}")
if not MODEL_PATH.is_dir():
    # Stop if the trained model cannot be found.
    raise FileNotFoundError(f"Saved model does not exist: {MODEL_PATH}")
if not LABELS_PATH.is_file():
    # Stop because numeric outputs need their class names.
    raise FileNotFoundError(f"Labels file does not exist: {LABELS_PATH}")

# Read the class names saved by build_image_model.py.
# Read the JSON text from disk.
labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
# Extract the list of class names.
class_names = labels_data.get("class_names")
if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
    # Stop instead of guessing the meaning of numeric outputs.
    raise ValueError(f"Invalid class names in {LABELS_PATH}")

# Load the saved model and its prediction entry point.
# Read the model directory from disk.
saved_model = tf.saved_model.load(str(MODEL_PATH))
# Keras normally uses the "serve" signature.
predict_function = saved_model.signatures.get("serve")
if predict_function is None:
    # Also support models saved with the older signature name.
    predict_function = saved_model.signatures.get("serving_default")
if predict_function is None:
    # Stop if the model has no callable prediction entry point.
    raise ValueError("Saved model does not contain a prediction signature.")

# Load and resize the image to the same size used during training.
image = tf.keras.utils.load_img(
    # File to load.
    IMAGE_PATH,
    # Resize to the dimensions used during training.
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # Load as RGB first because TensorFlow's BMP decoder may reject channels=1.
    color_mode="rgb",
)
# Convert the image object into numeric pixel values.
image_array = tf.keras.utils.img_to_array(image)
# Convert the decoded RGB pixels into one grayscale channel.
image_array = tf.image.rgb_to_grayscale(image_array)

# The model expects a batch. Add one image to a batch:
# (128, 128, 1) becomes (1, 128, 128, 1).
image_batch = tf.expand_dims(image_array, axis=0)

# Ask the model for one probability per class.
# Run the saved prediction function on the one-image batch.
prediction_outputs = predict_function(image_batch)
# Get the probability tensor returned by the model.
probabilities = next(iter(prediction_outputs.values())).numpy()[0]
# Find the position of the highest probability.
predicted_index = int(np.argmax(probabilities))
if predicted_index >= len(class_names):
    # Stop if the model and labels file do not describe the same classes.
    raise ValueError("The model output does not match the saved class names.")

# Convert the largest probability back into the matching folder name.
# Use the selected output position to get the class name.
predicted_class = class_names[predicted_index]
# Convert the selected probability to a normal Python number.
confidence = float(probabilities[predicted_index])
# Print the final result for the user.
print(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")
