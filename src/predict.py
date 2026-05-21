import joblib
import numpy as np


# Load saved artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict_stability(features):
    """
    Predict electrical grid stability.

    Parameters:
        features (list): Input feature values

    Returns:
        int: Prediction (0 or 1)
    """

    # Convert to NumPy array
    features_array = np.array(features).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features_array)

    # Predict
    prediction = model.predict(scaled_features)

    return int(prediction[0])


# Example test
if __name__ == "__main__":

    sample_input = [
        2.3, 3.1, 1.2, 4.5,
        0.5, 0.8, 0.6, 0.9,
        0.3, 0.2, 0.7, 0.4,
        1.1
    ]

    result = predict_stability(sample_input)

    print(f"Prediction: {result}")
