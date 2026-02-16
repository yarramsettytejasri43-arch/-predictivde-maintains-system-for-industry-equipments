from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df['stress_index'] = (
        df['temperature'] * 0.3 +
        df['vibration'] * 0.3 +
        df['pressure'] * 0.2 +
        df['load_percentage'] * 0.2
    )

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "failure_probability": float(probability)
    }
