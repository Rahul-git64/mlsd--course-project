from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np


# Initialize FastAPI app
app = FastAPI()


# Load saved model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Define request body schema
class InputData(BaseModel):
    features: list


# Home route
@app.get("/")
def home():
    return {"message": "Electrical Grid Stability Prediction API"}


# Prediction route
@app.post("/predict")
def predict(data: InputData):

    # Convert input to NumPy array
    input_array = np.array(data.features).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)

    # Convert prediction to label
    result = "stable" if prediction[0] == 1 else "unstable"

    return {
        "prediction": result
    }
