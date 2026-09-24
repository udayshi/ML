# Convolutional Neural Network: Jump-Start Guide

This project uses TensorFlow to learn the difference between image classes.

There are two scripts:

- `cnn_build.py` reads labeled images, trains a neural network, and saves it.
- `cnn_load.py` loads the saved network and predicts the class of one image.

## Setup with `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) to create the Python environment, install packages, and run scripts.

### 1. Install `uv`

### 2. Install the packages used by the Python files

The scripts import these third-party packages:

| Package | Used by | Purpose |
| --- | --- | --- |
| `tensorflow` | `cnn_build.py`, `cnn_load.py` | Load images, train/export the classifier, and run predictions |
| `numpy` | `cnn_load.py` | Select the prediction with the highest probability |

Install them into the `uv` environment:

```bash
uv add tensorflow numpy
```

The `json`, `pathlib`, and `__future__` imports are from Python's standard library and do not require separate packages.



## 1. The image folders

TensorFlow uses folder names as labels. The expected structure is:

```text
datasets/
├── training_set/
│   ├── cats/
│   ├── dogs/
│   └── rabbits/
└── test_set/
    ├── cats/
    ├── dogs/
    └── rabbits/
```

For example, an image inside `training_set/cats` receives the label `cats`, while an image inside
`training_set/rabbits` receives the label `rabbits`. You can add a new class by creating a matching folder in both datasets and placing images inside it.

The two dataset folders have different jobs:

- `training_set`: images used to teach the model.
- `test_set`: separate images used to check the model after training.

Using separate images for testing helps show whether the model can recognize new images instead of only remembering its training images.

## 2. What is a model?

A machine-learning model is a program with many adjustable numbers called **weights**. During training, TensorFlow changes those weights so the model learns visual patterns.

The model may learn patterns such as edges, shapes, eyes, ears, and textures. It does not think about a cat exactly as a person does; it learns numerical patterns that are common in the labeled examples.

## 3. How `cnn_build.py` works

The training script follows this process:

1. Read images from the folders.
2. Resize all images to the same dimensions.
3. Create a convolutional neural network.
4. Train the network using the labeled images.
5. Check it using the test images.
6. Save the trained model and all discovered class names.

### Imports and constants

The script imports `json` for saving labels, `Path` for file paths, and `tensorflow` for image processing and neural-network functionality. Everything runs from top to bottom; there are no user-defined functions to jump between.

The scripts do not use command-line parameters. Instead, edit the constants near the top of the file when you need different folders, image paths, or training settings.

These constants control the input data:

```python
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 1  # Grayscale: one brightness channel.
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIRECTORY = Path("datasets/training_set")
VALIDATION_DIRECTORY = Path("datasets/test_set")
MODEL_PATH = Path("models/cnn-demo.model")
```

Every image is changed to 128 pixels wide by 128 pixels high. The value `1` means the model uses grayscale, with one brightness channel instead of separate red, green, and blue channels. The files are first decoded as RGB because TensorFlow's BMP decoder does not support direct one-channel decoding; they are converted to grayscale immediately afterward. Grayscale reduces the model input data and can make training faster, although it may reduce accuracy when color is an important clue. A batch is a group of images processed together; this script uses 32 images per batch. `EPOCHS` controls how many times the model sees the complete training dataset.

### Loading the datasets

The script calls TensorFlow's image-folder loader:

```python
tf.keras.utils.image_dataset_from_directory(
    directory,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=shuffle,
)
```

TensorFlow finds the images, uses their folder names as labels, resizes them, and groups them into batches.

The training data is shuffled when it is loaded so the model does not always see images in the same order. The validation/test data is not shuffled because its order is not important. Since the dataset already handles shuffling, `model.fit` uses `shuffle=False` to avoid a warning and duplicate shuffle setting.

TensorFlow also returns the class names. They may look like:

```python
["cats", "dogs"]
```

Internally, the model works with numbers instead of words. For three folders, a possible mapping is:

```text
cats -> 0
dogs -> 1
rabbits -> 2
```

The script saves the actual discovered order in `labels.json`, so the prediction script can convert the number back into a readable name.

### Creating the neural network

The script creates a `tf.keras.Sequential` model. Sequential means that each image passes through the layers in order. The final layer receives the number of discovered class folders through `class_count`.

```python
Input(shape=(128, 128, 1))
```

This tells the model the expected shape of one image: height, width, and one grayscale channel.

```python
Rescaling(1.0 / 255.0)
```

Image pixels normally range from 0 to 255. This layer converts them to 0.0 to 1.0. Normalized numbers usually make neural-network training easier.

```python
Conv2D(32, 3, activation="relu")
MaxPooling2D()
```

`Conv2D` scans small areas of the image and learns visual features. Early convolution layers commonly learn simple edges and lines; later layers combine them into more useful shapes.

`MaxPooling2D` reduces the amount of image data while keeping strong features. This makes the model faster and less sensitive to the exact position of a feature.

The script uses three convolution layers with 32, 64, and 128 filters. A filter is a small pattern detector. More filters allow the network to detect more patterns.

```python
Flatten()
Dense(128, activation="relu")
Dense(class_count, activation="softmax")
```

`Flatten` changes the feature grids into one long list of numbers. `Dense` layers combine those features to make a decision.

The final layer has one output for every discovered class. For three classes, it has three outputs. `softmax` turns the outputs into probabilities, for example:

```text
cats: 0.15
dogs: 0.70
rabbits: 0.15
```

The probabilities add up to approximately 1.0.

### Compiling and training

Before training, the model is compiled:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

- `adam` decides how to adjust the model weights.
- `sparse_categorical_crossentropy` measures how wrong a prediction is when labels are numbers such as 0 and 1.
- `accuracy` tells TensorFlow to display the percentage of correct predictions.

Then this code starts training:

```python
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
)
```

One **epoch** means the model has processed the complete training dataset once. For every batch, TensorFlow makes predictions, compares them with the correct labels, calculates the error, and adjusts the weights.

The validation dataset is checked after each epoch. Too few epochs may mean the model has not learned enough. Too many epochs may cause **overfitting**, where the model remembers training images but performs badly on new images.

### Saving the result

The trained model is exported to:

```text
models/cnn-demo.model/
```

This is a TensorFlow SavedModel directory, even though its name ends with `.model`. It contains the neural-network structure, weights, and a prediction function.

The script also writes:

```text
models/cnn-demo.model/labels.json
```

For example:

```json
{
  "class_names": ["cats", "dogs"]
}
```

The model returns numbers, so this file tells the loader which text label belongs to each output position. This is why adding a new class does not require changing `cnn_load.py`.

## 4. How `cnn_load.py` works

The prediction script follows this process:

1. Check that the image and model exist.
2. Read the saved class names.
3. Load the TensorFlow SavedModel.
4. Load and resize the new image.
5. Convert the image to numbers and add a batch dimension.
6. Ask the model for probabilities.
7. Select the largest probability and print its label.

### Loading the model and labels

```python
saved_model = tf.saved_model.load(str(MODEL_PATH))
predict_function = saved_model.signatures.get("serve")
```

`tf.saved_model.load` reads the model created by the training script. The loader first looks for the `serve` signature and falls back to `serving_default`; either signature is the saved entry point that accepts images and returns predictions.

The label-loading code reads `labels.json`. If the file is missing or has the wrong format, the script raises a clear error instead of guessing the labels.

### Preparing one image

```python
image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
)
image_array = tf.keras.utils.img_to_array(image)
```

The image is resized to 128 by 128 and decoded as RGB. It is then converted to grayscale before being passed to the model. The resulting numerical array has this shape:

```text
128 x 128 x 1
```

The model was trained with batches, so the script adds a batch dimension:

```python
image_batch = tf.expand_dims(image_array, axis=0)
```

The final shape is:

```text
1 x 128 x 128 x 1
```

The first `1` means that the batch contains one image and the final `1` is the grayscale channel. The model's rescaling layer performs the same pixel normalization used during training.

### Selecting the prediction

The serving function may return probabilities such as:

```text
[0.91, 0.09]
```

The script uses NumPy's `argmax` to find the position of the largest number. If position 0 is largest, it uses the first label; if position 1 is largest, it uses the second label.

The final output looks like:

```text
Prediction: dogs (91.00% confidence)
```

Confidence is the model's probability estimate, not a guarantee that the answer is correct.

## 5. Running the scripts

Train the model from the project directory with `uv`:

```bash
uv run cnn_build.py
```

To change the training settings, edit these constants in `cnn_build.py`:

```python
TRAIN_DIRECTORY = Path("datasets/training_set")
VALIDATION_DIRECTORY = Path("datasets/test_set")
MODEL_PATH = Path("models/cnn-demo.model")
EPOCHS = 10
```

Classify the default image using the default model:

```bash
uv run cnn_load.py
```

To classify a different image or use a different model, edit these constants in `cnn_load.py`:

```python
IMAGE_PATH = Path("datasets/single_prediction/d-1.jpg")
MODEL_PATH = Path("models/cnn-demo.model")
```

If test files are added, run them with:

```bash
uv run pytest
```

This repository currently does not contain test files. A future test suite should check validation and label-loading behavior without training the complete model, because training can take time and depends on the computer's hardware.

## 6. Adding another class

To add a new class, for example `rabbits`:

1. Create `datasets/training_set/rabbits/`.
2. Add rabbit training images to that folder.
3. Create `datasets/test_set/rabbits/`.
4. Add separate rabbit test images to that folder.
5. Run `uv run cnn_build.py` again.

TensorFlow discovers all folder names, the final neural-network layer is sized automatically, and `labels.json` is regenerated with the new class.

The training and test directories must contain the same class folders. The model requires at least two classes.

## 7. Common questions

### Do I need to train every time?

No. Run `cnn_build.py` when you want to create or retrain a model. After that, use `cnn_load.py` for predictions.

### Why might the prediction be wrong?

The model may struggle with blurry or dark images, images containing both animals, drawings, stuffed animals, partly hidden animals, or images very different from the training examples.

More varied and balanced training images usually improve results.

### Why must the folder names match?

The folder names become the labels. Both training and test directories must contain the same matching class folders, such as `cats`, `dogs`, and `rabbits`.

## 8. Complete flow

```text
Labeled image folders
        |
        v
Resize images and create batches
        |
        v
Train convolutional neural network
        |
        v
Save model and labels.json
        |
        v
Load a new image
        |
        v
Resize and normalize it
        |
        v
Model produces probabilities
        |
        v
Print the predicted class with confidence
```
