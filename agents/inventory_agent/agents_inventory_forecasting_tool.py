# agents/inventory_forecasting_tool.py
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
import sqlite3
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import numpy as np

# --- Setup DB ---
db_path = os.path.abspath("db/erp.db")
db_uri = f"sqlite:///{db_path}"
db = SQLDatabase.from_uri(db_uri)

# --- LSTM Model ---
class DemandForecastLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(DemandForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Tool: Forecast Demand ---
@tool
def forecast_tool(sku: str, days: int = 30) -> str:
    """Forecasts demand using LSTM model trained on historical sales."""
    try:
        # 1. Load data from DB - FIXED query
        query = f"""
        SELECT o.created_at as date, SUM(oi.quantity) as quantity
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_id = p.id
        WHERE p.sku = '{sku}' AND o.status != 'cancelled'
        GROUP BY date(o.created_at)
        ORDER BY date(o.created_at)
        """
        
        conn = sqlite3.connect(db_path.replace("sqlite:///", ""))
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty or len(df) < 15:  # Need at least 15 days of data
            return f"❌ Not enough sales data found for {sku}. Need at least 15 days of historical data."

        # 2. Prepare time series data
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Fill missing dates with 0
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(date_range, fill_value=0)
        
        # Use last 90 days for training
        recent_data = df[-90:].copy()

        if len(recent_data) < 15:
            return f"❌ Not enough recent data for {sku}. Only {len(recent_data)} days available."

        # 3. Scale data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(recent_data[["quantity"]])

        # 4. Create sequences
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                y = data[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        seq_length = 14
        X, y = create_sequences(data, seq_length)

        if len(X) == 0:
            return f"❌ Not enough data to create sequences for {sku}"

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
        y = torch.tensor(y, dtype=torch.float32)

        # 5. Train simple model (fewer epochs for demo)
        model = DemandForecastLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train for fewer epochs
        epochs = 20
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 6. Forecast
        model.eval()
        last_seq = X[-1].unsqueeze(0)  # Shape: (1, seq_len, 1)
        forecasts = []
        
        with torch.no_grad():
            current_seq = last_seq.clone()
            for _ in range(days):
                pred = model(current_seq)
                forecasts.append(pred.item())
                # Update sequence: remove first, add prediction
                new_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)], dim=1)
                current_seq = new_seq

        # Inverse transform and format
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts)
        
        # Remove negative forecasts
        forecasts = np.maximum(forecasts, 0)
        total_forecast = int(np.sum(forecasts))
        
        daily_avg = int(np.mean(forecasts)) if len(forecasts) > 0 else 0
        
        return (f"📊 Forecast for {sku} over next {days} days:\n"
                f"• Total demand: {total_forecast} units\n"
                f"• Daily average: {daily_avg} units\n"
                f"• Peak daily: {int(np.max(forecasts))} units\n"
                f"(Using PyTorch LSTM on {len(recent_data)} days of historical data)")

    except Exception as e:
        return f"❌ Forecast failed: {str(e)}"

# Alternative simple forecasting for testing
@tool
def simple_forecast_tool(sku: str, days: int = 30) -> str:
    """Simple forecasting using historical averages (fallback)."""
    try:
        query = f"""
        SELECT AVG(oi.quantity) as avg_daily_sales
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_id = p.id
        WHERE p.sku = '{sku}' 
        AND o.created_at >= date('now', '-30 days')
        AND o.status != 'cancelled'
        """
        
        conn = sqlite3.connect(db_path.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        
        if not result or result[0] is None:
            return f"❌ No sales data found for {sku} in the last 30 days."
        
        avg_daily = result[0] or 0
        total_forecast = int(avg_daily * days)
        
        return f"📊 Simple forecast for {sku} over {days} days: {total_forecast} units (based on {avg_daily:.1f} units/day average)"

    except Exception as e:
        return f"❌ Simple forecast failed: {str(e)}"
    