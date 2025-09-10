# agents_finance_ML.py

import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
import os
from dotenv import load_dotenv  

load_dotenv()

# Optional: Log once that this is local ML
print("📊 Anomaly detection running locally with Isolation Forest (no LLM Needed)")

# --- Load Data from DB ---
def load_invoice_data(db_path: str):
    """Load unpaid invoices for anomaly detection."""
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT total_amount, 
               julianday('now') - julianday(issue_date) as days_old,
               customer_id,
               id as invoice_id,
               issue_date
        FROM invoices
        WHERE status = 'unpaid'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"❌ Error loading invoice data: {e}")
        return pd.DataFrame()

# --- Train Anomaly Model ---
def train_anomaly_model(db_path: str):
    """Train an Isolation Forest model on invoice features."""
    df = load_invoice_data(db_path)
    if df.empty:
        print("⚠️ No invoice data available for training.")
        return None

    X = df[["total_amount", "days_old"]]
    
    # Handle case where only one sample
    if len(X) == 1:
        print("⚠️ Only one invoice — skipping model training.")
        return None

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    print(f"✅ Model trained on {len(df)} unpaid invoices")
    return model

# --- Predict Anomalies ---
def detect_anomalies(db_path: str, model) -> list:
    """Detect and return anomalous unpaid invoices."""
    df = load_invoice_data(db_path)
    if df.empty or model is None:
        return []

    X = df[["total_amount", "days_old"]]

    # Handle single row case
    if len(X) == 1:
        pred = [model.predict(X)[0]]
    else:
        pred = model.predict(X)

    df["anomaly"] = pred
    anomalies = df[df["anomaly"] == -1]  # -1 = outlier
    print(f"🚨 Found {len(anomalies)} suspicious unpaid invoices")
    return anomalies.to_dict("records")