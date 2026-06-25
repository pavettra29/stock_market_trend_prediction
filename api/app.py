"""
api/app.py — Flask REST API for Stock Trend Prediction
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from predict import predict, compute_features, download_recent_data
from config import US_STOCKS, INDIAN_STOCKS, FEATURE_COLS, LOOKBACK
app = Flask(__name__)
CORS(app)
PHASE3_METRICS = {
    "AAPL": {"auc": 0.483, "accuracy": 0.50, "val_acc": 0.50},
    "MSFT": {"auc": 0.537, "accuracy": 0.52, "val_acc": 0.52},
    "GOOGL": {"auc": 0.499, "accuracy": 0.51, "val_acc": 0.51},
    "AMZN": {"auc": 0.470, "accuracy": 0.49, "val_acc": 0.49},
    "NVDA": {"auc": 0.566, "accuracy": 0.54, "val_acc": 0.53},
}
INDIAN_TO_US_PROXY = {
    "TCS": "MSFT",
    "RELIANCE": "AMZN",
    "HDFCBANK": "MSFT",
}
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})
@app.route("/api/predict")
def api_predict():
    ticker = request.args.get("ticker", "").upper().strip()
if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
try:
        result = predict(ticker, verbose=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/overview")
def api_overview():
    all_tickers = US_STOCKS + INDIAN_STOCKS
    results = []
    for ticker in all_tickers:
        try:
            r = predict(ticker, verbose=False)
            results.append(r)
        except Exception as e:
            results.append({
                "ticker": ticker,
                "direction": "ERROR",
                "probability": 0,
                "error": str(e),
                "proxy_used": ticker in INDIAN_STOCKS,
                "model_used": INDIAN_TO_US_PROXY.get(ticker, ticker),
            })
    return jsonify({
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
    })
@app.route("/api/chart")
def api_chart():
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    is_indian = ticker in INDIAN_STOCKS
    yf_ticker = (ticker + ".NS") if is_indian else ticker
    try:
        df = yf.download(yf_ticker, period="90d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        sma20 = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = sma20 + 2 * bb_std
        bb_lower = sma20 - 2 * bb_std

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

        dates = df.index.strftime("%Y-%m-%d").tolist()

        def clean(series):
            return [round(float(v), 4) if not np.isnan(v) else None for v in series.values]

        return jsonify({
            "ticker": ticker,
            "dates": dates,
            "close": clean(close),
            "high": clean(high),
            "low": clean(low),
            "volume": clean(volume),
            "sma20": clean(sma20),
            "bb_upper": clean(bb_upper),
            "bb_lower": clean(bb_lower),
            "rsi": clean(rsi),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/performance")
def api_performance():
    return jsonify({
        "metrics": PHASE3_METRICS,
        "description": "AUC and accuracy scores from Phase 3 test-set evaluation",
    })

# Production + Development entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n🚀 Flask API running at http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
