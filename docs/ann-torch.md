# PyTorch Artificial Neural Network: Jump-Start Guide

This example is the PyTorch version of the ANN workflow in [ann.md](ann.md). It predicts the binary `isActive` value from customer account information using the same CSV data, preprocessing steps, network shape, and interactive prediction flow.

## Install the packages

After initializing the uv project from the main [README](../README.md), install the packages used by this example:

```bash
uv add torch scikit-learn numpy pandas
```

- `torch` builds, trains, saves, and loads the neural network.
- `scikit-learn` encodes categories, splits the data, and scales features.
- `numpy` and `pandas` prepare the data and prediction input.

## 1. Generate the CSV dataset

Run the project data generator:

```bash
uv run dummy_csv.py
```

This creates `data/ann.csv` only when the file does not already exist. The ANN data contains these columns:

```text
CreditScore,Location,Gender,Age,RightsToWork,Balance,TotalProducts,HasCrCard,isActiveMember,isActive
895,Spain,F,27,0,69774,2,1,0,1
```

The first nine columns are input features. The final `isActive` column is the binary target.

## 2. Build and save the PyTorch ANN

Run:

```bash
uv run ann_build_torch.py
```

The script follows the same preparation flow as `ann_build.py`:

```python
dataset = pd.read_csv("data/ann.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

It then:

1. Converts `Gender` to numeric values with `LabelEncoder`.
2. One-hot encodes `Location` with `OneHotEncoder`.
3. Splits the rows into 80% training and 20% test data.
4. Fits `StandardScaler` on the training features.
5. Converts the training arrays to PyTorch tensors.

The four location values expand the nine original input features to 12 features.

### Model

The PyTorch model has the same shape as the TensorFlow model:

```python
ann = torch.nn.Sequential(
    torch.nn.Linear(12, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 1),
    torch.nn.Sigmoid(),
)
```

The training loop uses Adam, binary cross-entropy loss, and 100 epochs. The trained weights are saved to:

```text
models/ann_torch.pt
```

## 3. Load the model and make a prediction

After training, run:

```bash
uv run ann_load_torch.py
```

The loader creates the same 12-input model, loads `models/ann_torch.pt`, and switches it to evaluation mode. It then scales the same example input used by the original loader:

```python
new_data = np.array([[1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]])
```

The output has the same form as the TensorFlow version:

```text
Model loaded from ./models/ann_torch.pt
Prediction: [[0.48]]
Customer will Leave: [[False]]
```

The script waits for input after each prediction. Press `Ctrl+C` to stop it.

The output says `Customer will Leave`, but the training target is named `isActive`. Interpret this as a prediction for the target represented by the training data, not as a verified churn prediction.

## Important limitations

The generated features and `isActive` target are independently random, so they do not represent a meaningful relationship. The model's predictions may be weak or unstable and should not be used for real customer decisions.

The loader creates a new `StandardScaler` and calls `fit_transform` on the single prediction row, matching the original TensorFlow loader. For production inference, save the fitted scaler and encoders during training and use `transform` only when loading new data.

## Common problems

### `ModuleNotFoundError: No module named 'torch'`

Install PyTorch:

```bash
uv add torch
```

### The CSV file cannot be found

Run `uv run dummy_csv.py` from the project directory. It creates `data/ann.csv` when the file is missing.

### The model file cannot be found

Run `uv run ann_build_torch.py` before `uv run ann_load_torch.py`. The build script creates `models/ann_torch.pt`.

### The model input shape does not match

The location encoding produces 12 input features. New data must use the same 12-value order as the training data.

## Project files

```text
.
├── data/
│   └── ann.csv              # ANN input data
├── models/
│   └── ann_torch.pt        # Saved PyTorch model weights
├── ann_build_torch.py      # Builds, trains, and saves the ANN
├── ann_load_torch.py       # Loads the ANN and predicts
└── README.md               # Project overview and setup
```

## Suggested next steps

1. Save the fitted scaler and encoders with the model.
2. Rename the `Customer will Leave` message to match the `isActive` target.
3. Add classification metrics such as precision, recall, and AUC.
