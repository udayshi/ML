# Simple Linear Regression

This project is a small machine-learning example that trains a **simple linear regression** model. The model uses `ExperienceYears` to predict `Salary` from a CSV file.

It demonstrates a complete beginner-friendly workflow:

1. Install the Python dependencies.
2. Generate a CSV dataset.
3. Load the dataset with pandas.
4. Split the data into training and test sets.
5. Train a scikit-learn linear regression model.
6. Compare real salaries with predicted salaries.

## Quick start

From the project directory, install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv init
uv add scikit-learn numpy pandas matplotlib
```

Generate the sample dataset and run the regression example:

```bash
uv run dummy_csv.py
uv run simple-linear-regression.py
```

The first command creates `data/salary.csv`. The second command trains the model and prints actual salaries, predicted salaries, differences, and the model score.

## Documentation

For the full beginner walkthrough, including setup details, explanations, limitations, common problems, and suggested next steps, see:

**[Read the complete simple linear regression walkthrough](docs/simple-linear-regression.md)**

## Project files

```text
.
├── data/
│   └── salary.csv                 # Generated input data
├── docs/
│   └── simple-linear-regression.md # Detailed walkthrough
├── dummy_csv.py                   # Creates the sample CSV
├── simple-linear-regression.py    # Trains and evaluates the model
└── README.md                      # Project overview
```

> Note: The sample generator creates `ExperienceYears` and `Salary` independently using random values. This makes the project useful for learning the machine-learning workflow, but the resulting model should not be used for real salary predictions.
