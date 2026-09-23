# Machine Learning Regression Examples

This project contains small, beginner-friendly examples of regression with Python and scikit-learn. The examples use CSV files, pandas, NumPy, and `uv` so that each step of the machine-learning workflow is easy to inspect.

The project currently includes:

- **Simple linear regression** — predicts `Salary` from one input, `ExperienceYears`.
- **Multiple linear regression** — predicts `Profit` from several numeric inputs and a categorical `Location` value.

Both examples demonstrate how to load data, prepare features, split data into training and test sets, train a model, and compare predictions with known values.

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

## Run the examples

The dataset generator creates the CSV files used by both examples:

```bash
uv run dummy_csv.py
```

Then run either model:

```bash
uv run simple-linear-regression.py
uv run multiple-linear-regression.py
```

The generator currently writes random values, so the exact predictions and scores can change between runs. The data is intended for learning the workflow, not for real business or salary predictions.

## Project files

```text
.
├── data/
│   ├── salary.csv                    # Input data for simple regression
│   └── startup.csv                   # Input data for multiple regression
├── docs/
│   ├── simple-linear-regression.md   # Simple regression walkthrough
│   └── multiple-linear-regression.md # Multiple regression walkthrough
├── dummy_csv.py                      # Creates both sample CSV files
├── simple-linear-regression.py       # One-feature regression example
├── multiple-linear-regression.py     # Multi-feature regression example
└── README.md                         # Project overview and navigation
```
