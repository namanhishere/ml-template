import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.predictors import PREDICTORS

app = FastAPI(title="AI-ML-Template Serving", version="0.1.0")
predictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    ckpt_path = os.environ.get("CKPT_PATH", "checkpoints/best.pt")
    predictor = PREDICTORS.instantiate("classification", ckpt_path=ckpt_path, device="cuda:0" if torch.cuda.is_available() else "cpu")


class PredictRequest(BaseModel):
    inputs: list


class PredictResponse(BaseModel):
    predictions: list
    model_name: str
    framework_version: str


@app.get("/health")
def health():
    return {"status": "ok", "framework_version": "0.1.0", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    results = [predictor.predict(inp) for inp in req.inputs]
    return PredictResponse(predictions=results, model_name=predictor.model.__class__.__name__, framework_version="0.1.0")


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(req: PredictRequest):
    results = predictor.predict_batch(req.inputs)
    return PredictResponse(predictions=results, model_name=predictor.model.__class__.__name__, framework_version="0.1.0")
