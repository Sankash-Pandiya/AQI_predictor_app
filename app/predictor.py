"""
AQI Predictor — FastAPI inference service
Loads the trained sklearn pipeline (aqi_pipeline.pkl) and exposes:
  POST /predict   → next-day AQI prediction
  GET  /health    → liveness probe
  GET  /docs      → auto-generated Swagger UI
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.transformers import Winsorizer

# Ensure Winsorizer resolves if loaded from a pickled __main__ scope
sys.modules["__main__"].Winsorizer = Winsorizer

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "aqi_pipeline.pkl"))

_pipeline = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model on startup, release on shutdown."""
    global _pipeline
    if MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)
    else:
        # Allow startup even without model (tests inject a mock)
        _pipeline = None
    yield
    _pipeline = None


app = FastAPI(
    title="AQI Predictor API",
    description="Predicts next-day Air Quality Index using a trained ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    All fields that the pipeline expects.
    Missing sensor readings can be passed as null — the pipeline will impute them.
    """

    model_config = {"populate_by_name": True}

    City: str = Field(..., json_schema_extra={"example": "Delhi"})

    # Raw pollutant readings
    PM2_5: float | None = Field(None, alias="PM2.5")
    PM10: float | None = None
    NO: float | None = None
    NO2: float | None = None
    NOx: float | None = None
    NH3: float | None = None
    CO: float | None = None
    SO2: float | None = None
    O3: float | None = None
    Benzene: float | None = None
    Toluene: float | None = None

    # AQI history features
    AQI: float | None = Field(None, description="Today's observed AQI")
    aqi_lag1: float | None = None
    aqi_lag2: float | None = None
    aqi_lag3: float | None = None
    aqi_delta1: float | None = None
    aqi_roll7_mean: float | None = None
    aqi_roll14_mean: float | None = None
    aqi_roll30_mean: float | None = None
    aqi_roll7_std: float | None = None

    # Calendar features
    month_sin: float | None = None
    month_cos: float | None = None
    dayofweek: int | None = Field(None, ge=0, le=6)
    is_weekend: int | None = Field(None, ge=0, le=1)
    quarter: int | None = Field(None, ge=1, le=4)


class PredictionResponse(BaseModel):
    city: str
    predicted_next_day_aqi: float
    aqi_category: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AQI_BUCKETS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

FEATURE_COLUMNS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene",
    "AQI", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_delta1",
    "aqi_roll7_mean", "aqi_roll14_mean", "aqi_roll30_mean", "aqi_roll7_std",
    "month_sin", "month_cos", "dayofweek", "is_weekend", "quarter",
    "City",
]


def aqi_to_category(aqi: float) -> str:
    for lo, hi, label in AQI_BUCKETS:
        if lo <= aqi <= hi:
            return label
    return "Severe"


def request_to_df(req: PredictionRequest) -> pd.DataFrame:
    """Convert Pydantic request to a one-row DataFrame matching feature order."""
    row = {
        "PM2.5": req.PM2_5,
        "PM10": req.PM10,
        "NO": req.NO,
        "NO2": req.NO2,
        "NOx": req.NOx,
        "NH3": req.NH3,
        "CO": req.CO,
        "SO2": req.SO2,
        "O3": req.O3,
        "Benzene": req.Benzene,
        "Toluene": req.Toluene,
        "AQI": req.AQI,
        "aqi_lag1": req.aqi_lag1,
        "aqi_lag2": req.aqi_lag2,
        "aqi_lag3": req.aqi_lag3,
        "aqi_delta1": req.aqi_delta1,
        "aqi_roll7_mean": req.aqi_roll7_mean,
        "aqi_roll14_mean": req.aqi_roll14_mean,
        "aqi_roll30_mean": req.aqi_roll30_mean,
        "aqi_roll7_std": req.aqi_roll7_std,
        "month_sin": req.month_sin,
        "month_cos": req.month_cos,
        "dayofweek": req.dayofweek,
        "is_weekend": req.is_weekend,
        "quarter": req.quarter,
        "City": req.City,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_pipeline is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = request_to_df(req)
    raw = float(_pipeline.predict(df)[0])
    aqi = float(np.clip(raw, 0, 500))

    return PredictionResponse(
        city=req.City,
        predicted_next_day_aqi=round(aqi, 2),
        aqi_category=aqi_to_category(aqi),
    )
