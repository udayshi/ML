# Simple Linear Regression: Beginner Walkthrough

Simple linear regression uses one input feature to predict one target, making it a clear starting point for understanding the basic machine-learning workflow in this demo.

The project demonstrates a complete beginner-friendly workflow:

1. Install the Python dependencies.
2. Generate a CSV dataset.
3. Load the dataset with pandas.
4. Split the data into training and test sets.
5. Train a scikit-learn linear regression model.
6. Compare real salaries with predicted salaries.

## Install the packages

After initializing the uv project from the main [README](../README.md), install the packages used by this example:

```bash
uv add scikit-learn numpy pandas matplotlib
```

The code uses these Python packages:

- `scikit-learn` — machine-learning algorithms. The code imports it as `sklearn`.
- `numpy` — numerical arrays and calculations.
- `pandas` — reading and creating CSV data.
- `matplotlib` — included by the regression script for plotting support, although this version does not currently display a plot.

Although the Python import is named `sklearn`, the package you install is named `scikit-learn`:

```python
from sklearn.linear_model import LinearRegression
```

## 1. Generate the CSV dataset

Run the dataset generator:

```bash
uv run dummy_csv.py
```

This creates or replaces `data/salary.csv`. The file contains 100 rows with two columns:

```text
ExperienceYears,Salary
4,412110
5,238854
```

The values are randomly generated each time the script runs. Therefore, the exact rows and model results will change from run to run.

## 2. Run the linear regression example

After creating the CSV file, run:

```bash
uv run simple-linear-regression.py
```

The script performs the following actions:

```python
dataset = pd.read_csv("./data/salary.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here, `X` contains the input feature (`ExperienceYears`) and `y` contains the value to predict (`Salary`). The data is then split into training and testing portions:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0
)
```

The model learns from the training data:

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

Finally, it predicts the salaries in the test set and prints the real value, predicted value, difference, and model score.

The output is not fixed because `dummy_csv.py` generates new random data. A typical output has this shape:

```text
Index   Real    Predicted       Difference      Difference %
(0,)-   ...     ...             ...             ...%
...
Total   ...     with score ...
```

## Why use this approach?

This workflow is intentionally small because it makes the essential machine-learning pipeline visible:

- A CSV file represents a common real-world data source.
- pandas turns that file into data that Python can work with.
- NumPy provides the array operations used by the model and evaluation code.
- A train/test split checks the model on data it did not train on.
- scikit-learn provides a well-tested implementation of linear regression, so we can focus on understanding the workflow rather than implementing the mathematics from scratch.
- `uv run` ensures the script runs with the project dependencies, which makes the commands easier to reproduce on another machine.

This is useful for learning and for quickly checking an idea before building a larger application. In a production project, the dataset would normally come from a trusted source, the data would be validated, the random seed would be controlled for reproducible experiments, and the model would be evaluated with stronger metrics and visualizations.

## Important limitation of the sample data

The generator currently chooses `ExperienceYears` and `Salary` independently using random numbers. It does **not** create a real relationship between experience and salary. Consequently, the model score may be low or unstable, and this example should not be used to make salary predictions.

That limitation is useful pedagogically: it shows that a model can always produce predictions, but predictions are only meaningful when the input data contains a genuine, reliable relationship.

## Common problems

### `ModuleNotFoundError: No module named 'sklearn'`

Install the distribution package with its correct name:

```bash
uv add scikit-learn
```

### `ModuleNotFoundError: No module named 'pandas'`

The dataset generator and regression script both use pandas:

```bash
uv add pandas
```

### The regression script cannot find `data/salary.csv`

Run `uv run dummy_csv.py` first, and run both commands from the project directory. The generator creates the `data/salary.csv` file expected by the regression script.

### Results change every time

That is expected because the dataset uses random values. To make experiments repeatable, add a seed before generating the random columns, for example:

```python
np.random.seed(0)
```

## Project files

```text
.
├── data/
│   └── salary.csv                 # Generated input data
├── dummy_csv.py                   # Creates the sample CSV
├── simple-linear-regression.py    # Trains and evaluates the model
└── README.md                      # This walkthrough
```

## Suggested next steps

1. Add a random seed and compare two runs.
2. Change the generator so salary increases with experience.
3. Print the learned slope and intercept.
4. Add a scatter plot and regression line with matplotlib.
5. Add tests for CSV creation, input validation, and model evaluation.
