import streamlit as st
import pandas as pd
import requests
import os
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Finlora Fraud Detection", layout="centered")

# Reads from an environment variable when set (e.g. inside Docker, where
# 127.0.0.1 would point at the wrong container), falls back to localhost
# for running both processes directly on your own machine
API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000/predict")

CSV_PATH = Path(__file__).resolve().parent / "Finlora_Dataset" / "artifacts" / "Cleaned_Data.csv"

# Only currencies the live API's calculate_amount_usd actually supports —
# offering anything else here would guarantee a 400 error on submit
SUPPORTED_SOURCE_CURRENCIES = ["USD", "CAD", "GBP"]

@st.cache_data
def load_historical_data():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def preload_index(options, known_value):
    """
    Shared lookup for every dropdown that preloads a customer's known value.
    One function instead of copy-pasting this check for each field, so a
    future field added here can't silently skip the preload by accident.
    """
    return options.index(known_value) if known_value in options else 0

historical_data = load_historical_data()

channel_options = sorted(historical_data["channel"].dropna().unique().tolist())
kyc_tier_options = sorted(historical_data["kyc_tier"].dropna().unique().tolist())
home_country_options = sorted(historical_data["home_country"].dropna().unique().tolist())
ip_country_options = sorted(historical_data["ip_country"].dropna().unique().tolist())
dest_currency_options = sorted(historical_data["dest_currency"].dropna().unique().tolist())

st.title("Finlora Fraud Detection")
st.caption("Score a new transaction against the trained model.")

st.subheader("1. Customer")

customer_ids = sorted(historical_data["customer_id"].unique().tolist())
selected_customer = st.selectbox("Customer ID", customer_ids)

customer_rows = historical_data[historical_data["customer_id"] == selected_customer].sort_values("timestamp")
latest_known = customer_rows.iloc[-1]

st.caption(f"Loaded from {len(customer_rows)} known transaction(s), most recent: {latest_known['timestamp']}")

st.subheader("2. This Transaction")

col1, col2 = st.columns(2)

with col1:
    amount_src = st.number_input("Amount (source currency)", min_value=0.01, value=100.0, step=1.0)
    source_currency = st.selectbox("Source Currency", SUPPORTED_SOURCE_CURRENCIES)
    dest_currency = st.selectbox("Destination Currency", dest_currency_options)
    # Not preloaded from history, same as amount_src — fee is transaction-specific,
    # not a customer attribute, so it should describe THIS transaction, not the last one
    fee = st.number_input("Fee", min_value=0.0, value=0.0, step=0.5)
    channel = st.selectbox("Channel", channel_options)

with col2:
    home_country = st.selectbox("Home Country", home_country_options, index=preload_index(home_country_options, latest_known["home_country"]))
    ip_country = st.selectbox("IP Country", ip_country_options, index=preload_index(ip_country_options, latest_known["ip_country"]))
    kyc_tier = st.selectbox("KYC Tier", kyc_tier_options, index=preload_index(kyc_tier_options, latest_known["kyc_tier"]))
    new_device = st.selectbox("New Device?", ["No", "Yes"])
    location_mismatch = st.selectbox("Location Mismatch?", ["No", "Yes"])

st.subheader("3. Account Signals (preloaded, editable)")

col3, col4 = st.columns(2)

with col3:
    ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, float(latest_known["ip_risk_score"]))
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, float(latest_known["device_trust_score"]))
    account_age_days = st.number_input("Account Age (days)", min_value=0, value=int(latest_known["account_age_days"]))

with col4:
    chargeback_history_count = st.number_input("Chargeback History Count", min_value=0.0, value=float(latest_known["chargeback_history_count"]))
    risk_score_internal = st.slider("Internal Risk Score", 0.0, 1.0, float(latest_known["risk_score_internal"]))
    corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, float(latest_known["corridor_risk"]))

    st.subheader("4. Predict")

if st.button("Run Fraud Check", type="primary"):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": selected_customer,
        "home_country": home_country,
        "source_currency": source_currency,
        "dest_currency": dest_currency,
        "channel": channel,
        "amount_src": amount_src,
        "fee": fee,
        "new_device": new_device,
        "ip_country": ip_country,
        "location_mismatch": location_mismatch,
        "ip_risk_score": ip_risk_score,
        "kyc_tier": kyc_tier,
        "account_age_days": account_age_days,
        "device_trust_score": device_trust_score,
        "chargeback_history_count": chargeback_history_count,
        "risk_score_internal": risk_score_internal,
        "corridor_risk": corridor_risk,
    }

    try:
        with st.spinner("Scoring transaction..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
        st.session_state["last_result"] = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is your FastAPI server running?")
    except requests.exceptions.HTTPError:
        st.error(f"API returned an error: {response.status_code} — {response.text}")
    else:
        result = st.session_state["last_result"]

        st.subheader("Result")

        if result["is_fraud"] == 1:
            st.error(f"FLAGGED AS FRAUD — probability {result['fraud_probability']:.1%}")
        else:
            st.success(f"Not flagged — fraud probability {result['fraud_probability']:.1%}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Amount (USD)", f"${result['amount_usd']:.2f}")
        col6.metric("Velocity (1h)", result["txn_velocity_1h"])
        col7.metric("Velocity (24h)", result["txn_velocity_24h"])

        if result["velocity_spike"]:
            st.warning("Velocity spike detected on this account.")