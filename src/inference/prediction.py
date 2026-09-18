import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import OneHotEncoder

# Holds the full historical transaction dataset once loaded - used to compute
# velocity features (txn_velocity_1h/24h) for a customer at prediction time
historical_data = None

# In-memory cache of very recent transactions per customer, so velocity checks
# stay accurate between historical data loads (e.g. transactions submitted
# after the historical CSV was last generated)
transaction_cache = {}

# Static currency-to-USD conversion rates for live prediction. Training data
# had a real per-transaction exchange_rate_src_to_dest column; at prediction
# time we don't have that, so these are averages derived from our own
# Engineered_Data.csv (see Exploratory.ipynb for the derivation).
EXCHANGE_RATE = {
    'CAD': 0.739669,
    'GBP': 1.249187,
    'USD': 1.0
}


def load_historical_data(csv_path):
    """
    Load historical transaction data from CSV.
    Must be Cleaned_Data.csv — the only artifact with both customer_id and
    timestamp still intact (Engineered_Data.csv drops both).
    """
    global historical_data

    historical_data = pd.read_csv(csv_path)

    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

    # Remove timezone if present, so later comparisons with current_time
    # don't raise a tz-mismatch error
    if historical_data['timestamp'].dt.tz is not None:
        historical_data['timestamp'] = historical_data['timestamp'].dt.tz_localize(None)

    historical_data = historical_data.sort_values(['customer_id', 'timestamp'])
    print(f"loaded {len(historical_data)}")
    print(f"found {historical_data['customer_id'].nunique()} unique customer..")

    return historical_data


def add_transaction_to_cache(customer_id, timestamp, amount_src, amount_usd):
    """
    Add a new transaction to the cache for velocity spike signal.
    """
    global transaction_cache

    if customer_id not in transaction_cache:
        transaction_cache[customer_id] = []

    transaction_cache[customer_id].append({
        "timestamp": timestamp,
        "amount_src": amount_src,
        "amount_usd": amount_usd
    })

    # Cap cache size per customer to prevent unbounded memory growth
    if len(transaction_cache[customer_id]) > 100:
        transaction_cache[customer_id] = transaction_cache[customer_id][-100:]


def calculate_amount_usd(amount_src, source_currency):
    """
    Convert a source-currency amount to USD using our static rate table.
    Raises if the currency isn't recognized, rather than silently
    defaulting to a 1.0 rate.
    """
    if source_currency not in EXCHANGE_RATE:
        raise ValueError(f"Unsupported currency: {source_currency}")
    return amount_src * EXCHANGE_RATE[source_currency]


def get_velocity_for_customer(customer_id, current_time):
    """
    Compute txn_velocity_1h and txn_velocity_24h for a customer, combining
    historical data with any recent transactions still in the cache.
    """
    global historical_data, transaction_cache

    if historical_data is None:
        raise ValueError("historical data not loaded")

    # Normalize current_time to tz-naive to match historical_data's timestamps
    if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
        current_time = current_time.replace(tzinfo=None)

    customer_txns = historical_data[historical_data['customer_id'] == customer_id]

    historical_timestamps = customer_txns['timestamp'].tolist() if not customer_txns.empty else []
    cached_txns = transaction_cache.get(customer_id, [])
    cached_timestamps = [txn['timestamp'] for txn in cached_txns]

    all_timestamps = historical_timestamps + cached_timestamps

    if not all_timestamps:
        return 0, 0

    count_1h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=1) < ts <= current_time)
    count_24h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=24) < ts <= current_time)

    return count_1h, count_24h

def engineer_feature(data):
    global historical_data

    df = data.copy()

    df['location_mismatch'] = df['location_mismatch'].map({'Yes': True, 'No': False}).astype(bool)
    df['new_device'] = df['new_device'].map({'Yes': True, 'No': False}).astype(bool)

    if 'amount_usd' not in df.columns or df['amount_usd'].isna().sum():
        df['amount_usd'] = df.apply(
            lambda row: calculate_amount_usd(row['amount_src'], row['source_currency']),
            axis=1
        )

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    if 'customer_id' in df.columns:
        for idx, row in df.iterrows():
            customer_id = row['customer_id']
            txn_time = row['timestamp']
            count_1h, count_24h = get_velocity_for_customer(customer_id, txn_time)
            df.loc[idx, 'txn_velocity_1h'] = count_1h
            df.loc[idx, 'txn_velocity_24h'] = count_24h

    df['late_night_hours'] = ((df['hour'] >= 3) & (df['hour'] <= 8)).astype(int)
    df['high_ip_risk'] = (df['ip_risk_score'] > 0.8).astype(int)
    df['low_device_trust'] = (df['device_trust_score'] < 0.3).astype(int)
    df['new_account'] = ((df['account_age_days'] >= 30) & (df['account_age_days'] <= 90)).astype(int)
    df['very_new_account'] = (df['account_age_days'] < 30).astype(int)
    df['velocity_spike'] = (df['txn_velocity_1h'] >= 3).astype(int)

    return df


def predict_transaction(model, input_data):
    """
    Returns (prediction, prediction_probability, df_engineered) — the third
    value lets the caller read amount_usd/velocity fields back out for the
    API response without re-running feature engineering a second time.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    df_engineered = engineer_feature(df)

    prediction = model.predict(df_engineered)[0]
    prediction_probability = model.predict_proba(df_engineered)[0][1]

    return prediction, prediction_probability, df_engineered