from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "delivery_model.pkl", "rb") as file:
    package = pickle.load(file)

model = package["model"]
features = package["features"]

app = FastAPI(title="Food Delivery Time Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class DeliveryInput(BaseModel):
    distance_km: float
    preparation_time_min: int
    traffic_level: int
    weather_severity: int
    rider_experience_years: int


@app.get("/")
def home():
    return {"message": "Food Delivery Prediction API is running"}


@app.post("/predict")
def predict(data: DeliveryInput):
    values = data.model_dump()
    new_delivery = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features,
    )

    prediction = float(model.predict(new_delivery)[0])

    return {"predicted_minutes": max(0, prediction)}