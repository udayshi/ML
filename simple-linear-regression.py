# NumPy provides numerical helpers, including the array iterator used below.
import numpy as np
# Matplotlib is available for future charts, although this script does not plot yet.
import matplotlib.pyplot as plt
# pandas loads the CSV file into a table-like DataFrame.
import pandas as pd
# This helper divides data into training and testing subsets.
from sklearn.model_selection import train_test_split
# LinearRegression is the machine-learning model used in this example.
from sklearn.linear_model import LinearRegression

# Read the generated salary data from the CSV file into a pandas DataFrame.
dataset = pd.read_csv('./data/salary.csv')
# Select every column except the last one as the input feature(s), X.
# In this dataset, X contains ExperienceYears.
X = dataset.iloc[:, :-1].values
# Select the last column as the target value, y.
# In this dataset, y contains Salary, which the model will learn to predict.
y = dataset.iloc[:, -1].values


# Split the data into training and testing sets.
# The model learns from two-thirds of the rows and is evaluated on the remaining third.
# random_state=0 makes the split repeatable even though the generated CSV can change.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
regressor = LinearRegression()

# Fit the model by learning the relationship between the training features and salaries.
regressor.fit(X_train, y_train)

# Use the trained model to predict salaries for the unseen test features.
y_pred = regressor.predict(X_test)

# Calculate the model's R² score on the test data.
# A score closer to 1 generally means the predictions explain more of the target variation.
score=regressor.score(X_test, y_test)

# Print the headings for the row-by-row evaluation output.
print(f"Index\tReal\tPredicted\tDifference\tDifference %")
# Keep a running total of the percentage differences.
percent:float=0
# Keep a running total of the prediction differences.
total_diff_pred:float=0

# Iterate through each real salary in the test set and its array index.
for k,v in np.ndenumerate(y_test):
    # Store the actual salary from the test dataset.
    real_value:float=v
    # Use the matching index to retrieve the model's predicted salary.
    predicted_value:float=y_pred[k]
    # Calculate the signed difference: actual value minus predicted value.
    diff:float=real_value-predicted_value
    # Express the difference as a percentage of the actual salary.
    diff_percentage:float=(diff/real_value)*100
    # Add this row's percentage difference to the running total.
    percent+=diff_percentage
    # Add this row's signed difference to the running total.
    total_diff_pred+=diff

    # Print the index, actual salary, prediction, difference, and percentage difference.
    print(f"{k}- \t{real_value:.2f}\t{predicted_value:.2f}\t{diff:.2f}\t{diff_percentage:.2f}%")


# Print the average signed prediction difference and the overall R² score.
print(f"Total\t{total_diff_pred/len(y_test):.2f}  with score {score:.2f}")
