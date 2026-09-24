# Machine Learning Regression Examples

This project contains small, beginner-friendly examples of regression with Python and scikit-learn. The examples use CSV files, pandas, NumPy, and `uv` so that each step of the machine-learning workflow is easy to inspect.

The project currently includes:

- **Simple linear regression** — predicts `Salary` from one input, `ExperienceYears`.
- **Multiple linear regression** — predicts `Profit` from several numeric inputs and a categorical `Location` value.
- **Artificial neural network** — predicts the binary `isActive` value from customer account features.

These examples demonstrate how to load data, prepare features, split data into training and test sets, train a model, and evaluate or inspect predictions.

## Why I created this repository

I created this repository while preparing to work more closely with my company's data team. I wanted to learn some of the terminology used in machine learning and understand how the different pieces fit together in practice. I completed the 40-hour Udemy course **Machine Learning A-Z Python** and used this repository to reinforce the concepts with small, runnable examples. I am not an ML expert, but I now have a better understanding of how the basic components connect, and I hope these notes and examples can help others who are starting their own learning journey.

## Set up the uv project

Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). From this project directory, initialize the project once:

```bash
uv init
```

The package installation command belongs to each individual walkthrough because each document lists the dependencies needed by its example. Open the relevant guide below and follow its installation step before running the script.

## Choose a walkthrough

- [Simple linear regression](docs/simple-linear-regression.md) — use `ExperienceYears` to predict `Salary`.
- [Multiple linear regression](docs/multiple-linear-regression.md) — use administration, research, and marketing expenses plus location to predict startup profit.
- [Artificial neural network](docs/ann.md) — encode customer features and train a TensorFlow/Keras binary classifier.

## Run the examples

The dataset generator creates each missing CSV file used by the examples:

```bash
uv run dummy_csv.py
```

It creates `data/salary.csv`, `data/startup.csv`, and `data/ann.csv` only when those files do not already exist. See the [ANN walkthrough](docs/ann.md) for the TensorFlow dependency, model-building command, and model-loading command.

Then run the desired example (run `ann_build.py` before `ann_load.py` for the ANN):

```bash
uv run simple-linear-regression.py
uv run multiple-linear-regression.py
uv run ann_build.py
uv run ann_load.py
```

The generator writes random values when a file is first created, so the exact predictions and scores can vary. The ANN example saves its model to `models/ann.keras`. The data and models are intended for learning the workflow, not for real business, salary, or customer decisions.

## Project files

```text
.
├── data/
│   ├── salary.csv                    # Input data for simple regression
│   ├── startup.csv                   # Input data for multiple regression
│   └── ann.csv                       # Input data for the ANN classifier
├── docs/
│   ├── simple-linear-regression.md   # Simple regression walkthrough
│   ├── multiple-linear-regression.md # Multiple regression walkthrough
│   └── ann.md                        # ANN walkthrough
├── models/
│   └── ann.keras                     # Saved ANN model after training
├── dummy_csv.py                      # Creates missing sample CSV files
├── simple-linear-regression.py       # One-feature regression example
├── multiple-linear-regression.py     # Multi-feature regression example
├── ann_build.py                      # Builds and saves the ANN
├── ann_load.py                       # Loads the ANN and predicts
└── README.md                         # Project overview and navigation
```
