import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model
ann = tf.keras.models.load_model('./models/ann.keras')
print("Model loaded from ./models/ann.keras")

# Create a scaler instance (same parameters as training)
# Note: In production, you should save and load the scaler as well
sc = StandardScaler()

# Make prediction on new data
# Input: [1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]
new_data = np.array([[1, 0, 0, 600, 1, 20, 3, 60000, 2, 1, 1, 50000]])
scaled_data = sc.fit_transform(new_data)
while True:
    prediction = ann.predict(scaled_data,verbose=0)
    print(f"Prediction: {prediction}")
    print(f"Customer will Leave: {prediction > 0.5}")
    input(">")
