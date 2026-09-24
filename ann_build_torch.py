"""Build and save the PyTorch version of the ANN example."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load and preprocess data
dataset = pd.read_csv("data/ann.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical features for gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encode the location column
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train).astype(np.float32)
X_test = sc.transform(X_test).astype(np.float32)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

# Build model
ann = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 1),
    torch.nn.Sigmoid(),
)

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(ann.parameters())

# Train model
ann.train()
for epoch in range(100):
    optimizer.zero_grad()
    prediction = ann(X_train_tensor)
    loss = loss_function(prediction, y_train_tensor)
    loss.backward()
    optimizer.step()

# Save model
os.makedirs("./models", exist_ok=True)
torch.save(ann.state_dict(), "./models/ann_torch.pt")
print("Model saved to ./models/ann_torch.pt")
