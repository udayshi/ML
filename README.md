# Machine Learning Examples

This project contains small, jump-start machine-learning examples using Python, TensorFlow, PyTorch, scikit-learn, and `uv`. They cover regression, binary classification, image classification, and next-word prediction so that each workflow is easy to inspect.

The project currently includes:

- **Simple linear regression** — predicts `Salary` from one input, `ExperienceYears`.
- **Multiple linear regression** — predicts `Profit` from several numeric inputs and a categorical `Location` value.
- **Artificial neural network** — predicts the binary `isActive` value from customer account features.
- **PyTorch artificial neural network** — runs the same binary classification workflow using PyTorch.
- **Convolutional neural network** — classifies images using labeled folders and TensorFlow.
- **PyTorch convolutional neural network** — runs the same image-classification workflow using PyTorch.
- **Next-word prediction** — trains a TensorFlow/Keras LSTM from text files and generates likely following words.

These examples demonstrate how to load data, prepare features, split data into training and test sets, train a model, and evaluate or inspect predictions.

## Why I created this repository

I created this repository while preparing to work more closely with my company's data team. Even though I consider myself an AI expert and am programming-language agnostic, I am not an ML expert. I wanted to learn some of the terminology used in machine learning and understand how the different pieces fit together in practice. I completed the 40-hour Udemy course **Machine Learning A-Z Python** and used this repository to reinforce the concepts with small, runnable examples. I now have a better understanding of how the basic components connect, and I hope these notes and examples can help others getting started with machine learning.

## Set up the uv project

Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). From this project directory, initialize the project once:

```bash
uv init
```

The package installation command belongs to each individual walkthrough because each document lists the dependencies needed by its example. Open the relevant guide below and follow its installation step before running the script.

## Choose a guide

- [Simple linear regression](docs/simple-linear-regression.md) — use `ExperienceYears` to predict `Salary`.
- [Multiple linear regression](docs/multiple-linear-regression.md) — use administration, research, and marketing expenses plus location to predict startup profit.
- [Artificial neural network](docs/ann.md) — encode customer features and train a TensorFlow/Keras binary classifier.
- [PyTorch ANN](docs/ann-torch.md) — build and load the equivalent binary classifier with PyTorch.
- [Convolutional neural network](docs/CNN.md) — train and use an image classifier with TensorFlow/Keras.
- [PyTorch CNN](docs/cnn-torch.md) — train and use the equivalent image classifier with PyTorch.
- [Next-word prediction](docs/word-predict.md) — train an LSTM on text files and generate likely next words.

## Run the examples

The dataset generator creates each missing CSV file used by the examples:

```bash
uv run dummy_csv.py
```

It creates `data/salary.csv`, `data/startup.csv`, and `data/ann.csv` only when those files do not already exist. The next-word example instead reads existing `.txt` files below `data/`; its sample corpus is in `data/predict/`. See the relevant guide for framework-specific dependencies and commands.

Then run the desired example (run each build script before its matching load script):

```bash
uv run simple-linear-regression.py
uv run multiple-linear-regression.py
uv run ann_build.py
uv run ann_load.py
uv run ann_build_torch.py
uv run ann_load_torch.py
uv run cnn_build.py
uv run cnn_load.py
uv run cnn_build_torch.py
uv run cnn_load_torch.py
uv run word_predict_build.py
uv run word_predict.py
```

The generator writes random values when a file is first created, so the exact predictions and scores can vary. The TensorFlow ANN saves its model to `models/ann.keras`, the PyTorch ANN saves its checkpoint to `models/ann_torch.pt`, the PyTorch CNN saves its checkpoint to `models/cnn-demo-torch.pt`, and the next-word example saves `udays.keras` plus `tokenizer.pkl` in the project directory. The data and models are intended for learning the workflow, not for real business, salary, customer, or language-generation decisions.

## Project files

```text
.
├── data/
│   ├── salary.csv                    # Input data for simple regression
│   ├── startup.csv                   # Input data for multiple regression
│   ├── ann.csv                       # Input data for the ANN classifier
│   └── predict/                      # Text corpus for next-word prediction
├── docs/
│   ├── simple-linear-regression.md   # Simple regression walkthrough
│   ├── multiple-linear-regression.md # Multiple regression walkthrough
│   ├── ann.md                        # ANN walkthrough
│   ├── ann-torch.md                  # PyTorch ANN guide
│   ├── CNN.md                        # CNN walkthrough
│   ├── cnn-torch.md                  # PyTorch CNN guide
│   └── word-predict.md               # Next-word prediction walkthrough
├── models/
│   ├── ann.keras                     # Saved ANN model after training
│   ├── ann_torch.pt                  # Saved PyTorch ANN checkpoint
│   ├── cnn-demo.model/               # Exported CNN model after training
│   └── cnn-demo-torch.pt             # Saved PyTorch CNN checkpoint
├── dummy_csv.py                      # Creates missing sample CSV files
├── simple-linear-regression.py       # One-feature regression example
├── multiple-linear-regression.py     # Multi-feature regression example
├── ann_build.py                      # Builds and saves the ANN
├── ann_load.py                       # Loads the ANN and predicts
├── ann_build_torch.py                # Builds and saves the PyTorch ANN
├── ann_load_torch.py                 # Loads the PyTorch ANN and predicts
├── cnn_build.py                      # Builds and exports the CNN
├── cnn_load.py                       # Loads the CNN and classifies an image
├── cnn_build_torch.py                # Builds and saves the PyTorch CNN
├── cnn_load_torch.py                 # Loads the PyTorch CNN and classifies an image
├── word_predict_build.py              # Builds and saves the next-word model
├── word_predict.py                    # Generates words with the saved model
├── udays.keras                        # Saved next-word model after training
├── tokenizer.pkl                      # Saved next-word tokenizer after training
└── README.md                         # Project overview and navigation
```
