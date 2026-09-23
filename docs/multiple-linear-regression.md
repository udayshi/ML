# Multiple Linear Regression: Beginner Walkthrough

This project is a small machine-learning example that trains a **multiple linear regression** model. The model uses several startup expenses and a company's `Location` to predict `Profit` from a CSV file.

The project demonstrates a complete beginner-friendly workflow:

1. Install the Python dependencies.
2. Generate a CSV dataset.
3. Load the dataset with pandas.
4. Convert text categories into numbers.
5. Split the data into training and test sets.
6. Train a scikit-learn multiple linear regression model.
7. Compare predicted profits with the real profits.

## Install the packages

After initializing the uv project from the main [README](../README.md), install the packages used by this example:

```bash
uv add scikit-learn numpy pandas
```

The code uses these Python packages:

- `scikit-learn` — the machine-learning algorithms and preprocessing tools. The code imports it as `sklearn`.
- `numpy` — numerical arrays and the final prediction comparison.
- `pandas` — reading the startup data from a CSV file.

Although the Python import is named `sklearn`, the package you install is named `scikit-learn`:

```python
from sklearn.linear_model import LinearRegression
```

## 1. Generate the CSV dataset

From the project directory, run the dataset generator:

```bash
uv run dummy_csv.py
```

This creates or replaces `data/startup.csv`. The file contains 100 rows with these columns:

```text
AdminExp,RnDExp,MarketingExp,Location,Profit
373738,341281,336983,North London,327067
```

The first three columns are numeric inputs. `Location` is text and contains values such as `Central London`, `North London`, `South London`, and `East London`. `Profit` is the value the model will predict.

The values are randomly generated each time the script runs. Therefore, the exact rows and model results will change from run to run.

## 2. Run the multiple linear regression example

After creating the CSV file, run:

```bash
uv run multiple-linear-regression.py
```

The script reads the data and separates the input columns from the target column:

```python
dataset = pd.read_csv('./data/startup.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here, `X` contains `AdminExp`, `RnDExp`, `MarketingExp`, and `Location`. `y` contains `Profit`.

## 3. Convert the location text into numbers

Machine-learning models work with numerical values, so the text in `Location` must be encoded before training:

```python
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))
```

The location is column index `3`, so `OneHotEncoder` creates a separate indicator column for each location. For example, a row might receive a `1` in the `North London` column and `0` in the other location columns.

`remainder='passthrough'` keeps the three numeric expense columns unchanged. The transformed array therefore contains encoded location values alongside the original expenses.

## 4. Split and train the model

The data is split into training and testing portions:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

The model learns from 80% of the rows and is evaluated on the remaining 20%. `random_state=0` makes the split repeatable, although the generated CSV itself changes unless the random generator is seeded.

The regression model is then trained:

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

After training, it predicts profits for the test rows:

```python
y_pred = regressor.predict(X_test)
```

## 5. Read the output

The script formats the predictions, real values, and signed differences into columns:

```python
np.set_printoptions(precision=2)
pred = y_pred.reshape(len(y_pred), 1)
test = y_test.reshape(len(y_test), 1)
diff = test - pred
print(np.concatenate((pred, test, diff), 1))
```

The printed columns are:

1. Predicted profit
2. Real profit from the test data
3. Difference between real and predicted profit

The output is not fixed because `dummy_csv.py` generates new random data. A typical output has this shape:

```text
[[ predicted_profit  real_profit  difference ]
 ...]
```

## Why use this approach?

Multiple linear regression is useful when a target may depend on several inputs at the same time. This example makes that idea visible:

- A CSV file represents a common real-world data source.
- pandas turns the file into data that Python can work with.
- One-hot encoding converts a category such as `Location` into model-friendly numeric columns.
- Multiple numeric features allow the model to consider several possible influences on profit.
- A train/test split checks the model on data it did not train on.
- scikit-learn provides reliable preprocessing and regression implementations, so we can focus on understanding the workflow rather than implementing the mathematics from scratch.

This approach is useful for learning and for creating a quick baseline before trying more advanced models. In a production project, the data would normally come from a trusted source, input values would be validated, the preprocessing step would be saved with the model, and the model would be evaluated with stronger metrics and cross-validation.

## Important limitation of the sample data

The generator currently chooses expenses, location, and profit independently using random values. It does **not** create a genuine relationship between the inputs and profit. Consequently, the predictions and model quality may be weak or unstable, and this example should not be used to make business decisions.

That limitation is useful pedagogically: a model can produce predictions even when the data does not contain a meaningful pattern. Good predictions require relevant, reliable data.

## Common problems

### `ModuleNotFoundError: No module named 'sklearn'`

Install the distribution package with its correct name:

```bash
uv add scikit-learn
```

### `ModuleNotFoundError: No module named 'pandas'`

The regression script reads the CSV with pandas:

```bash
uv add pandas
```

### The regression script cannot find `data/startup.csv`

Run `uv run dummy_csv.py` first, and run both commands from the project directory. The generator creates the `data/startup.csv` file expected by the regression script.

### Results change every time

That is expected because the generator uses random values. To make experiments repeatable, add a seed before generating the random columns, for example:

```python
np.random.seed(0)
```

## Project files

```text
.
├── data/
│   └── startup.csv                # Generated input data
├── dummy_csv.py                   # Creates the startup CSV
├── multiple-linear-regression.py  # Trains and evaluates the model
└── README.md                      # Project overview and setup
```

## Suggested next steps

1. Add a random seed and compare two runs.
2. Print the model coefficients and intercept.
3. Add an R² score and mean absolute error calculation.
4. Compare predictions with a chart.
5. Add tests for CSV creation, category encoding, and model evaluation.
