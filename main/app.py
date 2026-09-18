from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from src.inference.prediction import (load_historical_data,
                                       predict_transaction,
                                       add_transaction_to_cache,
                                       calculate_amount_usd)

from src.utility.model_loader import load_registered_model

model = None
historical_data = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, historical_data

    try:
        model = load_registered_model()
        logger.info("model loaded successfully")
    except Exception:
        logger.exception("error occurred while loading the model")
        model = None

    try:
        # Path built relative to this file's own location, not hardcoded —
        # works regardless of which machine or working directory the
        # server is started from
        BASE_DIR = Path(__file__).resolve().parent.parent
        csv_path = BASE_DIR / "Finlora_Dataset" / "artifacts" / "Cleaned_Data.csv"

        historical_data = load_historical_data(csv_path)
        logger.info("historical data has been successfully created")
    except Exception:
        logger.exception("error occurred during loading of historical dataset")
        historical_data = None

    yield  # server runs while paused here; nothing needed on shutdown for this project


app = FastAPI(title="fraud detection API", lifespan=lifespan)


class TransactionData(BaseModel):
    timestamp: str
    customer_id: str
    home_country: str
    source_currency: str
    dest_currency: str
    channel: str
    amount_src: float
    fee: float
    new_device: Optional[str] = "No"
    ip_country: str
    location_mismatch: Optional[str] = "No"
    ip_risk_score: float
    kyc_tier: str
    account_age_days: int
    device_trust_score: float
    chargeback_history_count: float
    risk_score_internal: float
    corridor_risk: float


# what the API gives back as its response
class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    txn_velocity_1h: Optional[int] = None
    txn_velocity_24h: Optional[int] = None
    velocity_spike: Optional[int] = None
    amount_usd: Optional[float] = None


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionData):  # sync, not async — predict_transaction is CPU-bound
    global model, historical_data

    if model is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    if historical_data is None:
        raise HTTPException(status_code=500, detail="historical data has not been loaded")

    try:
        # pydantic v2 — model_dump(), not .to_dict() (not a real method) or v1's .dict()
        input_data = transaction.model_dump()

        # predict_transaction engineers features internally and returns df_engineered too,
        # so we don't need to build/re-engineer a second dataframe just for the response
        prediction, prediction_probability, df_engineered = predict_transaction(model, input_data)
        amount_usd = float(df_engineered.iloc[0]['amount_usd'])

        # logged after prediction so this transaction isn't counted in its own velocity calc
        add_transaction_to_cache(
            transaction.customer_id,
            pd.to_datetime(transaction.timestamp),
            transaction.amount_src,
            amount_usd
        )

        return PredictionResponse(
            is_fraud=int(prediction),
            fraud_probability=float(prediction_probability),
            txn_velocity_1h=int(df_engineered.iloc[0]['txn_velocity_1h']),
            txn_velocity_24h=int(df_engineered.iloc[0]['txn_velocity_24h']),
            velocity_spike=int(df_engineered.iloc[0]['velocity_spike']),
            amount_usd=amount_usd,
        )

    except ValueError as e:
        # expected, deliberate input errors (e.g. unsupported currency) — safe to return to caller
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        # anything else is a server-side problem, not the client's fault —
        # log full detail server-side, don't leak internals to the API caller
        logger.exception("Unhandled error in /predict")
        raise HTTPException(status_code=500, detail="Internal error while processing prediction")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "historical_data_loaded": historical_data is not None
    }