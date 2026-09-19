# Fraudulent_Transaction_Detection_for_Finlora
This project is a Machine Learning Application that solves the issues of fraud transactions that occurs concurrently in Finlora's Application.

## API

The fraud prediction service is a FastAPI app in `main/app.py`.

### POST /predict

Scores a single transaction and returns a fraud prediction.

Request body, all fields required unless noted:

```json
{
  "timestamp": "2026-09-18T10:00:00",
  "customer_id": "example-customer-id",
  "home_country": "us",
  "source_currency": "USD",
  "dest_currency": "GBP",
  "channel": "web",
  "amount_src": 100.0,
  "fee": 2.5,
  "new_device": "No",
  "ip_country": "us",
  "location_mismatch": "No",
  "ip_risk_score": 0.2,
  "kyc_tier": "standard",
  "account_age_days": 365,
  "device_trust_score": 0.8,
  "chargeback_history_count": 0.0,
  "risk_score_internal": 0.3,
  "corridor_risk": 0.2
}
```

`new_device` and `location_mismatch` default to `"No"` if omitted.

Response body:

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0045,
  "txn_velocity_1h": 0,
  "txn_velocity_24h": 1,
  "velocity_spike": 0,
  "amount_usd": 100.0
}
```

Returns 400 if `source_currency` is not one of the supported currencies. Returns 422 automatically if a required field is missing or the wrong type. Returns 500 if the model or historical data failed to load at startup.

### GET /health

Returns service status, including whether the model and historical data loaded successfully at startup.

## Running locally

```bash
uvicorn main.app:app --reload
```

## Running with Docker

Build the image from the project root:

```bash
docker build -t finlora-fraud-api .
```

The model is loaded from the MLflow Model Registry hosted on DagsHub, so the container needs tracking credentials passed in at runtime rather than baked into the image:

```bash
docker run -p 8000:8000 -e MLFLOW_TRACKING_USERNAME=<your_dagshub_username> -e MLFLOW_TRACKING_PASSWORD=<your_dagshub_token> finlora-fraud-api
```

Then test it:

```bash
curl http://127.0.0.1:8000/health
```
