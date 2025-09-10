import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# --- Load Data from DB ---
def load_invoice_data(db_path: str):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT total_amount, 
           julianday('now') - julianday(issue_date) as days_old,
           customer_id
    FROM invoices
    WHERE status = 'unpaid'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# --- Train Anomaly Model ---
def train_anomaly_model(db_path: str):
    df = load_invoice_data(db_path)
    if df.empty:
        return None
    X = df[["total_amount", "days_old"]]
    model = IsolationForest(contamination=0.1)
    model.fit(X)
    return model

# --- Predict Anomalies ---
def detect_anomalies(db_path: str, model) -> list:
    df = load_invoice_data(db_path)
    if df.empty or model is None:
        return []
    X = df[["total_amount", "days_old"]]
    df["anomaly"] = model.predict(X)
    return df[df["anomaly"] == -1].to_dict("records")