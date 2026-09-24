"""Load the PyTorch ANN example and predict one customer record."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Build the same model used during training
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

# Load the saved model
ann.load_state_dict(torch.load("./models/ann_torch.pt", map_location="cpu"))
ann.eval()
print("Model loaded from ./models/ann_torch.pt")

# Create a scaler instance (same parameters as training)
# Note: In production, save and load the fitted scaler as well.
sc = StandardScaler()

# Make prediction on new data
# Input: [1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]
new_data = np.array([[1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]])
scaled_data = sc.fit_transform(new_data).astype(np.float32)
scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

while True:
    with torch.no_grad():
        prediction = ann(scaled_data_tensor).numpy()
    print(f"Prediction: {prediction}")
    print(f"Customer will Leave: {prediction > 0.5}")
    input(">")
