import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Finlora Fraud Detection", layout="wide")

# Reads from an environment variable when set (e.g. inside Docker, where
# 127.0.0.1 would point at the wrong container), falls back to localhost
# for running both processes directly on your own machine
API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000/predict")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "Finlora_Dataset" / "artifacts"
CSV_PATH = ARTIFACTS_DIR / "Cleaned_Data.csv"
DASHBOARD_CSV_PATH = ARTIFACTS_DIR / "test_predictions.csv"
FLAGGED_SHAP_CSV_PATH = ARTIFACTS_DIR / "flagged_shap_values.csv"
MODEL_METRICS_CSV_PATH = ARTIFACTS_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_CSV_PATH = ARTIFACTS_DIR / "shap_feature_importance.csv"

# Only currencies the live API's calculate_amount_usd actually supports,
# offering anything else here would guarantee a 400 error on submit
SUPPORTED_SOURCE_CURRENCIES = ["USD", "CAD", "GBP"]

# Colours are reused consistently across every dashboard chart so the same
# meaning always maps to the same colour: red always means fraud or a
# feature pushing toward fraud, slate always means genuine, blue always
# means a feature pushing away from fraud
COLOR_GENUINE = "#64748B"
COLOR_FRAUD = "#DC2626"
COLOR_TOWARD_FRAUD = "#DC2626"
COLOR_TOWARD_GENUINE = "#2563EB"
COLOR_IMPORTANCE = "#2563EB"


@st.cache_data
def load_historical_data():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_dashboard_data():
    df = pd.read_csv(DASHBOARD_CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_flagged_shap():
    df = pd.read_csv(FLAGGED_SHAP_CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_model_metrics():
    return pd.read_csv(MODEL_METRICS_CSV_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_CSV_PATH)


def preload_index(options, known_value):
    """
    Shared lookup for every dropdown that preloads a customer's known value.
    One function instead of copy-pasting this check for each field, so a
    future field added here can't silently skip the preload by accident.
    """
    return options.index(known_value) if known_value in options else 0


tab_dashboard, tab_score = st.tabs(["Fraud Intelligence Dashboard", "Score a Transaction"])

with tab_dashboard:
    dashboard_data = load_dashboard_data()
    importance_df = load_feature_importance()

    st.title("Finlora Fraud Intelligence Dashboard")
    st.caption(
        "Built from the model's held-out test set, 1,790 transactions the model "
        "never trained on, so every number here reflects genuine out-of-sample performance."
    )

    # 1. Transaction overview
    st.subheader("1. Transaction Overview")

    total_transactions = len(dashboard_data)
    actual_fraud_count = int(dashboard_data["actual_fraud"].sum())
    flagged_count = int(dashboard_data["predicted_fraud"].sum())
    avg_amount = dashboard_data["amount_usd"].mean()
    date_min = dashboard_data["timestamp"].min()
    date_max = dashboard_data["timestamp"].max()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Test Transactions", f"{total_transactions:,}")
    col_b.metric("Actual Fraud Rate", f"{actual_fraud_count / total_transactions:.1%}")
    col_c.metric("Flagged by Model", f"{flagged_count / total_transactions:.1%}")
    col_d.metric("Avg Amount (USD)", f"${avg_amount:,.2f}")

    st.caption(
        f"Test set period: {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}, "
        f"split from training at the customer level so no customer appears in both."
    )

    # 2. Fraud and legitimate transaction distributions
    st.subheader("2. Fraud vs Legitimate Transactions")

    dist_counts = (
        dashboard_data["actual_fraud"]
        .map({0: "Genuine", 1: "Fraud"})
        .value_counts()
        .reindex(["Genuine", "Fraud"])
        .reset_index()
    )
    dist_counts.columns = ["label", "count"]

    fig_dist = px.bar(
        dist_counts,
        x="label",
        y="count",
        color="label",
        color_discrete_map={"Genuine": COLOR_GENUINE, "Fraud": COLOR_FRAUD},
        text="count",
    )
    fig_dist.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Transactions")
    fig_dist.update_traces(textposition="outside")
    st.plotly_chart(fig_dist, use_container_width=True)

    # 3. Important fraud patterns
    st.subheader("3. Fraud Patterns: Key Risk Signals")
    st.caption(
        "The six features the model itself relies on most, ranked by mean absolute "
        "SHAP value, compared between genuine and fraudulent transactions."
    )

    pattern_features = [f for f in importance_df["feature"] if f in dashboard_data.columns][:6]

    pattern_rows = []
    for feat in pattern_features:
        for label, mask in [("Genuine", dashboard_data["actual_fraud"] == 0), ("Fraud", dashboard_data["actual_fraud"] == 1)]:
            pattern_rows.append({"feature": feat, "group": label, "mean_value": dashboard_data.loc[mask, feat].mean()})
    pattern_df = pd.DataFrame(pattern_rows)

    fig_pattern = px.bar(
        pattern_df,
        x="feature",
        y="mean_value",
        color="group",
        barmode="group",
        color_discrete_map={"Genuine": COLOR_GENUINE, "Fraud": COLOR_FRAUD},
    )
    fig_pattern.update_layout(xaxis_title=None, yaxis_title="Mean value", legend_title=None)
    st.plotly_chart(fig_pattern, use_container_width=True)

    # 4. Flagged transaction view
    st.subheader("4. Flagged Transactions")

    flagged_data = dashboard_data[dashboard_data["predicted_fraud"] == 1].sort_values(
        "fraud_probability", ascending=False
    )

    min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.05, key="flagged_min_prob")
    filtered_flagged = flagged_data[flagged_data["fraud_probability"] >= min_prob]

    st.caption(f"Showing {len(filtered_flagged)} of {len(flagged_data)} transactions the model flagged as fraud.")

    display_cols = [
        "customer_id", "timestamp", "fraud_probability", "channel", "kyc_tier",
        "home_country", "ip_country", "velocity_spike", "ip_risk_score",
    ]
    st.dataframe(
        filtered_flagged[display_cols].style.format({"fraud_probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )

    # 5. Model evaluation metrics
    st.subheader("5. Model Evaluation Metrics")

    metrics_df = load_model_metrics()
    st.dataframe(
        metrics_df.style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}", "roc_auc": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    metrics_long = metrics_df.melt(id_vars="model", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score")
    fig_metrics = px.bar(
        metrics_long,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_metrics.update_layout(xaxis_title=None, yaxis_title="Score", legend_title=None)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # 6. Feature importance visualisation
    st.subheader("6. Feature Importance (SHAP)")

    top_importance = importance_df.head(15).sort_values("mean_abs_shap")
    fig_importance = px.bar(
        top_importance,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLOR_IMPORTANCE],
    )
    fig_importance.update_layout(yaxis_title=None, xaxis_title="Mean absolute SHAP value")
    st.plotly_chart(fig_importance, use_container_width=True)

    # 7. Individual SHAP explanation
    st.subheader("7. Explain a Flagged Transaction")

    flagged_shap = load_flagged_shap()

    if flagged_shap.empty:
        st.info("No flagged transactions available to explain.")
    else:
        option_labels = (
            flagged_shap["customer_id"].astype(str)
            + ", "
            + flagged_shap["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            + ", probability "
            + (flagged_shap["fraud_probability"] * 100).round(1).astype(str)
            + "%"
        )
        selected_label = st.selectbox("Select a flagged transaction", option_labels)
        selected_row = flagged_shap.loc[option_labels == selected_label].iloc[0]

        feature_cols = [c for c in flagged_shap.columns if c not in ("customer_id", "timestamp", "fraud_probability")]
        contributions = selected_row[feature_cols].astype(float).sort_values(key=lambda s: s.abs()).tail(10)

        contrib_df = contributions.reset_index()
        contrib_df.columns = ["feature", "shap_value"]
        contrib_df["direction"] = contrib_df["shap_value"].apply(
            lambda v: "Pushes toward fraud" if v > 0 else "Pushes toward genuine"
        )

        fig_shap = px.bar(
            contrib_df,
            x="shap_value",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Pushes toward fraud": COLOR_TOWARD_FRAUD, "Pushes toward genuine": COLOR_TOWARD_GENUINE},
        )
        fig_shap.update_layout(yaxis_title=None, xaxis_title="SHAP value", legend_title=None)
        st.plotly_chart(fig_shap, use_container_width=True)

with tab_score:
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
        # Not preloaded from history, same as amount_src, fee is transaction-specific,
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
            st.error(f"API returned an error: {response.status_code}, {response.text}")
        else:
            result = st.session_state["last_result"]

            st.subheader("Result")

            if result["is_fraud"] == 1:
                st.error(f"FLAGGED AS FRAUD, probability {result['fraud_probability']:.1%}")
            else:
                st.success(f"Not flagged, fraud probability {result['fraud_probability']:.1%}")

            col5, col6, col7 = st.columns(3)
            col5.metric("Amount (USD)", f"${result['amount_usd']:.2f}")
            col6.metric("Velocity (1h)", result["txn_velocity_1h"])
            col7.metric("Velocity (24h)", result["txn_velocity_24h"])

            if result["velocity_spike"]:
                st.warning("Velocity spike detected on this account.")