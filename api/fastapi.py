import os
import torch
import joblib
from pydantic import BaseModel
from fastapi import FastAPI

MODEL_SCALER_PATH = os.getenv("MODEL_SCALER_PATH", "./data/model/scaler.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "./data/model/model.pth")

scaler = joblib.load(MODEL_SCALER_PATH)
mean = scaler["mean"]
std = scaler["std"]

model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))

app = FastAPI()

class PredictRequest(BaseModel):
    w1: float
    w2: float
    w3: float
    w4: float

@app.post("/predict")
def predict(request: PredictRequest):
    X = torch.tensor([request.w1, request.w2, request.w3, request.w4]).reshape(1, 4)
    X = (X - mean) / std

    y_pred = model(X).argmax(dim=1).item()
    return {"pred": y_pred}