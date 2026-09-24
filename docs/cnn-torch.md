# PyTorch Convolutional Neural Network: Line-by-Line Guide

This guide explains the PyTorch version of the image-classification workflow in
[CNN.md](CNN.md). The two scripts keep the same basic flow:

- cnn_build_torch.py loads labelled folders, trains a CNN, and saves it.
- cnn_load_torch.py loads the saved CNN and predicts image classes.

The example is intentionally small. It demonstrates the workflow rather than
providing a production training pipeline.

## Setup with uv

Install the required packages from the project directory:

~~~bash
uv add torch torchvision pillow
~~~

| Package | Used by | Purpose |
| --- | --- | --- |
| torch | Both scripts | Tensors, layers, training, and model weights |
| torchvision | Both scripts | Folder-based image loading and transformations |
| pillow | The loader and torchvision | Reading image files |

The json, pathlib, and shutil imports are from Python's standard library.

## 1. Prepare the image folders

PyTorch's ImageFolder uses each subfolder name as a class label. The training
and validation folders must use the same class folders:

~~~text
datasets/
├── training_set/
│   ├── Cat/
│   └── Dog/
├── test_set/
│   ├── Cat/
│   └── Dog/
└── single_prediction/
    ├── Cat/
    └── Dog/
~~~

The data_set.py helper copies a deterministic subset from ../datasets/set:

~~~bash
python3 data_set.py
~~~

For each class it copies 100 images to training_set, 20 to test_set, and 10
to single_prediction. The copied files are renamed sequentially, starting at 1,
so destinations contain names such as 1.jpg, 2.jpg, and 3.jpg.

The folders have these jobs:

- training_set: images used to update model weights.
- test_set: separate images used to measure validation accuracy.
- single_prediction: individual images selected for prediction.

## 2. What is a PyTorch model?

A model is a collection of layers containing adjustable numbers called
**weights**. During training, PyTorch changes those weights so the network
learns visual patterns from labelled images.

The model may learn patterns such as edges, shapes, eyes, ears, and textures.
It learns numerical patterns from the examples; it does not understand a cat or
dog as a person does.

## 3. How cnn_build_torch.py works

The build script follows this sequence:

1. Check that the training and validation folders exist.
2. Define one image transformation pipeline.
3. Load images and labels with ImageFolder.
4. Verify that the class folders match.
5. Group images into batches with DataLoader.
6. Create the convolutional neural network.
7. Train with Adam and cross-entropy loss.
8. Measure validation accuracy after every epoch.
9. Save the weights and class names.

### Module imports

~~~python
"""Train and save the PyTorch image classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
~~~

The module docstring describes the script. json saves the class names and Path
represents files and directories.

torch provides tensors and the optimizer. torch.nn provides layers and loss
functions. DataLoader creates batches. ImageFolder loads labelled folders, and
transforms provides preprocessing operations.

### Settings

~~~python
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIRECTORY = Path("datasets/training_set")
VALIDATION_DIRECTORY = Path("datasets/test_set")
MODEL_PATH = Path("models/cnn-demo-torch.pt")
LABELS_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.labels.json")
~~~

Each image is resized to 128 by 128 pixels. One channel means grayscale rather
than separate red, green, and blue channels.

A batch contains 32 images. EPOCHS controls how many times the model sees the
complete training dataset. The directory constants select the input data. The
weights and labels are saved under models/.

Edit these constants when using a different dataset or training configuration.

### Checking directories

~~~python
if not TRAIN_DIRECTORY.is_dir():
    raise FileNotFoundError(
        f"Training directory does not exist: {TRAIN_DIRECTORY}"
    )
~~~

The script stops early with a clear error when the training directory is
missing. It performs the same check for the validation directory before loading
validation images.

### Defining image transformations

~~~python
image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
        transforms.ToTensor(),
    ]
)
~~~

Compose applies the operations in order:

1. Resize changes every image to 128 by 128 pixels.
2. Grayscale changes every image to one brightness channel.
3. ToTensor converts the image to a PyTorch tensor and scales typical pixel
   values from 0–255 to approximately 0.0–1.0.

The resulting PyTorch image layout is:

~~~text
channels x height x width
1 x 128 x 128
~~~

PyTorch uses channels first, unlike the channel-last layout commonly used by
TensorFlow/Keras.

### Loading labelled datasets

~~~python
train_dataset = datasets.ImageFolder(
    TRAIN_DIRECTORY,
    transform=image_transform,
)
class_names = train_dataset.classes
class_count = len(class_names)
~~~

ImageFolder scans the subfolders and assigns each class a number. For example:

~~~text
Cat -> 0
Dog -> 1
~~~

class_names stores names in numeric-index order. The model trains with numbers,
while the saved JSON later converts an output number back to a class name.

~~~python
if class_count < 2:
    raise ValueError(
        f"Expected at least two class folders, received {class_count}."
    )
~~~

A classifier needs at least two possible outputs, so the script rejects a
single-class dataset.

The verbose messages show the discovered setup:

~~~python
print(f"Classes found ({class_count}): {class_names}")
print(f"Class mapping: {train_dataset.class_to_idx}")
~~~

This helps identify a missing or incorrectly named class before training starts.

### Loading validation data

~~~python
validation_dataset = datasets.ImageFolder(
    VALIDATION_DIRECTORY,
    transform=image_transform,
)
if validation_dataset.classes != class_names:
    raise ValueError(
        "Training and validation directories must contain the same class folders."
    )
~~~

Validation images use the same transformation pipeline. Their class folders
must match the training folders so numeric labels have the same meaning.

### Creating batches

~~~python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
~~~

The training loader shuffles images so the model does not always see them in the
same order. The validation loader does not shuffle because order does not affect
the accuracy calculation.

The script also prints dataset and batch counts:

~~~python
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches per epoch: {len(validation_loader)}")
~~~

### Creating the neural network

~~~python
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
~~~

Sequential sends each image through the layers from top to bottom.

- Conv2d scans small image areas and learns feature maps.
- ReLU adds a non-linear activation by replacing negative values with zero.
- MaxPool2d reduces height and width while keeping strong features.
- Flatten changes feature maps into one long vector.
- Linear layers combine features to produce class scores.

The three convolution layers use 32, 64, and 128 filters. With a 128 × 128
input, the final pooling layer produces approximately 128 × 14 × 14 values.
That is why the first linear layer has 128 * 14 * 14 inputs.

The final layer has one output for each discovered class. It returns raw scores,
called logits. There is no Softmax layer here because CrossEntropyLoss expects
logits. Softmax is used only by the loader when readable probabilities are
needed.

The script prints the architecture and parameter count:

~~~python
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print("Model architecture:")
print(model)
print(f"Trainable parameters: {parameter_count:,}")
~~~

### Configuring training

~~~python
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
~~~

CrossEntropyLoss compares raw class scores with integer labels such as 0 and 1.
Adam adjusts the model weights after each training batch.

### Training batches

~~~python
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    training_loss = 0.0
~~~

The outer loop repeats training for every epoch. model.train() selects training
mode. training_loss collects batch losses for the verbose summary.

~~~python
for images, labels in train_loader:
    optimizer.zero_grad()
    prediction = model(images)
    loss = loss_function(prediction, labels)
    loss.backward()
    optimizer.step()
    training_loss += loss.item()
~~~

Each batch follows this sequence:

1. zero_grad clears gradients from the previous batch.
2. The model produces raw predictions.
3. The loss compares predictions with the correct labels.
4. backward calculates gradients for the weights.
5. step updates weights using Adam.
6. loss.item() converts the loss to a Python number for logging.

### Measuring validation accuracy

~~~python
model.eval()
validation_correct = 0
validation_total = 0
with torch.no_grad():
    for images, labels in validation_loader:
        prediction = model(images)
        validation_correct += int(
            (prediction.argmax(dim=1) == labels).sum().item()
        )
        validation_total += labels.size(0)
~~~

model.eval() selects evaluation mode. torch.no_grad() prevents gradient storage
because validation does not update weights. argmax selects the output index with
the largest score for each image.

~~~python
validation_accuracy = validation_correct / validation_total
average_training_loss = training_loss / len(train_loader)
print(
    f"Epoch {epoch + 1}/{EPOCHS} - "
    f"training loss: {average_training_loss:.4f}, "
    f"validation accuracy: {validation_accuracy:.2%}"
)
~~~

The output summarizes learning:

~~~text
Epoch 1/100 - training loss: 0.6931, validation accuracy: 52.50%
~~~

Training loss describes the fit to training batches. Validation accuracy describes
performance on separate images. A large difference can indicate overfitting.

### Saving weights and labels

~~~python
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
LABELS_PATH.write_text(
    json.dumps({"class_names": class_names}, indent=2),
    encoding="utf-8",
)
~~~

The parent directory is created if necessary. state_dict() contains learned
weights rather than the complete Python model object.

Labels are saved separately because numeric output positions are meaningless
without the class order:

~~~text
models/cnn-demo-torch.pt
models/cnn-demo-torch.labels.json
~~~

## 4. How cnn_load_torch.py works

The loader performs these steps:

1. Check the image, weights, and labels files.
2. Read the saved class names.
3. Build the same network shape.
4. Load the saved weights.
5. Resize and grayscale each image.
6. Add a batch dimension.
7. Convert scores to probabilities.
8. Print the class with the highest probability.

### Loader settings and file checks

~~~python
MODEL_PATH = Path("models/cnn-demo-torch.pt")
IMAGE_PATH = Path("datasets/single_prediction/cat/10.jpg")
LABELS_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.labels.json")
~~~

MODEL_PATH and LABELS_PATH must match the files created by the build script.
IMAGE_PATH identifies the configured prediction image.

The script checks that the image, model, and labels file exist before continuing.
The current loader then loops through ten dog images, so update that loop when
using a different class or prediction directory.

### Reading labels

~~~python
labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
class_names = labels_data.get("class_names")
if not isinstance(class_names, list) or not all(
    isinstance(name, str) for name in class_names
):
    raise ValueError(f"Invalid class names in {LABELS_PATH}")
~~~

The JSON is parsed and the class list is validated. This prevents the loader from
guessing what a numeric model output means.

### Rebuilding the model

The loader creates the same convolution, pooling, flatten, and linear layers as
the build script. The final output size comes from the saved labels:

~~~python
torch.nn.Linear(128, len(class_names))
~~~

The number of output units must match the number of saved classes.

~~~python
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()
~~~

torch.load reads the weights and maps them to the CPU. load_state_dict places
them into the newly created model. eval() switches it to prediction mode.

### Preparing an image

~~~python
image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
        transforms.ToTensor(),
    ]
)

image = Image.open(IMAGE_PATH)
image_batch = image_transform(image).unsqueeze(0)
~~~

The loader applies the same preprocessing used during training. unsqueeze(0)
adds a batch dimension.

The actual PyTorch tensor shape is:

~~~text
batch x channels x height x width
1 x 1 x 128 x 128
~~~

### Selecting a prediction

~~~python
with torch.no_grad():
    probabilities = torch.softmax(model(image_batch), dim=1)[0]
predicted_index = int(torch.argmax(probabilities).item())
~~~

The model returns logits. softmax turns them into values that add up to
approximately 1.0 across classes. [0] selects the only image in the batch.
argmax selects the position of the largest probability.

The result is converted to a label and printed:

~~~python
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index].item())
print(f"Prediction: {predicted_class} ({confidence:.2%} confidence) id:{id}")
~~~

Confidence is the model's probability estimate, not a guarantee that the
prediction is correct. The current loop prints ten predictions and includes the
image id.

## 5. Running the scripts

From the project directory:

~~~bash
python3 data_set.py
uv run cnn_build_torch.py
uv run cnn_load_torch.py
~~~

If folder capitalization or the prediction path differs from the current
constants, update IMAGE_PATH and the prediction loop in cnn_load_torch.py.

## 6. Adding another class

To add a class such as Rabbit:

1. Add Rabbit to CLASS_NAMES in data_set.py if it exists in the source.
2. Create matching Rabbit folders under training_set and test_set.
3. Add labelled images to those folders.
4. Run cnn_build_torch.py again.

ImageFolder discovers the folders, and class_count automatically sizes the final
layer. The labels JSON is regenerated with the new class order.

The training and validation folders must contain exactly the same classes. The
classifier requires at least two classes.

## 7. Important limitations

This is a quick-start example:

- It saves only model weights, not a fitted preprocessing object.
- It always loads and predicts on the CPU.
- It does not use data augmentation.
- It does not include early stopping or checkpoint selection.
- It does not calculate precision, recall, or a confusion matrix.
- The sample dataset may not represent a meaningful real-world task.

For production use, preserve the complete preprocessing configuration, use a
separate evaluation set, and validate the model with task-appropriate metrics.

## 8. Complete flow

~~~text
Source image folders
        |
        v
Copy and rename a labelled subset
        |
        v
Resize images and convert them to grayscale tensors
        |
        v
Create batches with ImageFolder and DataLoader
        |
        v
Train the convolutional neural network
        |
        v
Measure validation accuracy
        |
        v
Save model weights and class names
        |
        v
Load and transform a new image
        |
        v
Convert logits to probabilities
        |
        v
Print the predicted class and confidence
~~~
