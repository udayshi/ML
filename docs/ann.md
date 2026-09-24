# Artificial Neural Network: Beginner Walkthrough

This example uses an artificial neural network (ANN) to predict the binary `isActive` value from customer account information. It demonstrates a complete beginner-friendly workflow:

1. Install the Python dependencies.
2. Generate a CSV dataset.
3. Load the dataset with pandas.
4. Encode categorical features.
5. Split and scale the data.
6. Build and train a TensorFlow/Keras neural network.
7. Save the trained model and load it for a prediction.

## Install the packages

After initializing the uv project from the main [README](../README.md), install the packages used by this example:

```bash
uv add tensorflow scikit-learn numpy pandas
```

The code uses these Python packages:

- `tensorflow` — builds, trains, saves, and loads the Keras neural network.
- `scikit-learn` — encodes categories, scales features, and splits the data. The code imports it as `sklearn`.
- `numpy` — stores and transforms the model inputs.
- `pandas` — reads and creates the CSV data.

## 1. Generate the CSV datasets

From the project directory, run:

```bash
uv run dummy_csv.py
```

The generator checks each file independently and creates it only when it does not already exist:

- `data/salary.csv` for simple linear regression.
- `data/startup.csv` for multiple linear regression.
- `data/ann.csv` for this ANN example.

The ANN dataset contains 100 rows with these columns:

```text
CreditScore,Location,Gender,Age,RightsToWork,Balance,TotalProducts,HasCrCard,isActiveMember,isActive
895,Spain,F,27,0,69774,2,1,0,1
```

The first nine columns are input features. The final column, `isActive`, is the binary value the network learns to predict. The values are randomly generated when the file is first created, so the exact rows and model results depend on the existing CSV file.

Because the generator does not replace an existing file, run the ANN example against the current `data/ann.csv` unless you remove or rename that file before generating a new sample dataset.

## 2. Build and save the ANN

Run the training script:

```bash
uv run ann_build.py
```

The script reads the data and separates the features from the target:

```python
dataset = pd.read_csv('data/ann.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here, `X` contains customer information and `y` contains `isActive`.

### Encode categorical features

The `Gender` column is converted to numeric labels with `LabelEncoder`. The `Location` column is converted into one-hot encoded columns with `OneHotEncoder`. This avoids treating locations as if they had a numeric order. The remaining numeric columns pass through unchanged.

### Split and scale the data

The data is split into 80% training rows and 20% test rows:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

`StandardScaler` learns the scaling values from `X_train` and applies the same transformation to `X_test`:

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Build and train the network

The model is a sequential network with three hidden layers and one output unit:

```python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

The final sigmoid unit produces a value between 0 and 1. The model uses the Adam optimizer, binary cross-entropy loss, batches of 32 rows, and 100 training epochs:

```python
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

The trained network is saved to `models/ann.keras`.

The script currently creates the `models` directory with `os.mkdir`, so running it again when that directory already exists raises `FileExistsError`. The first training run therefore needs the directory not to exist; later retraining requires the directory-creation behavior to be adjusted.

## 3. Load the model and make a prediction

After training successfully, run:

```bash
uv run ann_load.py
```

The loader reads the saved Keras model:

```python
ann = tf.keras.models.load_model('./models/ann.keras')
```

It then creates one example input with 12 values. The 12 values correspond to the nine original features after `Location` has been expanded into four one-hot columns:

```python
new_data = np.array([[1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]])
```

The model output is printed as a probability-like value and as a Boolean comparison. The script runs inside a loop and waits for input after each prediction; press `Ctrl+C` to stop it.

The output currently says `Customer will Leave`, but the training target is named `isActive`. Interpret the output as a prediction for the target represented by the training data, not as a verified churn prediction.

## Why use this approach?

This example introduces the main parts of a neural-network classification workflow:

- A CSV file represents a common real-world data source.
- Categorical values must be converted into numeric values before training.
- Scaling keeps numeric features on comparable ranges.
- Hidden layers learn combinations of the input features.
- A sigmoid output is suitable for a two-class prediction.
- Saving the model makes it possible to load the trained network later.

In a production system, the preprocessing encoders and scaler would be saved together with the model, the input schema would be validated, and the model would be evaluated with precision, recall, and AUC as well as accuracy.

## Important limitations of the sample data and loader

The generator chooses the customer features and `isActive` independently using random values. It does **not** create a meaningful relationship between the inputs and the target. The model's accuracy and predictions may therefore be weak or unstable, and this example should not be used for real customer decisions.

The loader also creates a new `StandardScaler` and calls `fit_transform` on the single prediction row. That is not the same scaling used during training. For reliable inference, save the fitted training scaler and category encoders, then apply them to new data with `transform` only.

## Common problems

### `ModuleNotFoundError: No module named 'tensorflow'`

Install TensorFlow for the current Python environment:

```bash
uv add tensorflow
```

### The script cannot find `data/ann.csv`

Run `uv run dummy_csv.py` first, and run the commands from the project directory. The generator creates the file when it is missing.

### `FileExistsError: [Errno 17] File exists: './models'`

`ann_build.py` currently calls `os.mkdir('./models')` on every run. The first training run creates the directory; later runs need the directory-creation behavior adjusted before retraining.

### The model input shape does not match

The training script one-hot encodes four possible `Location` values, so the model receives 12 features. A new input must use the same column order and encoding as the training data.

## Project files

```text
.
├── data/
│   └── ann.csv          # Generated ANN input data
├── models/
│   └── ann.keras        # Saved trained model
├── dummy_csv.py         # Creates the sample CSV files when missing
├── ann_build.py         # Builds, trains, and saves the ANN
├── ann_load.py          # Loads the ANN and prints predictions
└── README.md            # Project overview and setup
```

## Suggested next steps

1. Save the fitted scaler, label encoder, and one-hot encoder with the model.
2. Rename or correct the loader's `Customer will Leave` message to match `isActive`.
3. Add a validation set and print a confusion matrix, precision, recall, and AUC.
4. Add a random seed and generate a target with a meaningful relationship to the features.
5. Make the model directory creation safe for repeated training runs.
