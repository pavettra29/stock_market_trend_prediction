"""
Stock Trend Prediction API - Simplified for Deployment
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

from predict import predict
from config import US_STOCKS, INDIAN_STOCKS, MODEL_DIR

app = Flask(__name__)
CORS(app)

print(f"✅ MODEL_DIR = {MODEL_DIR}")
print(f"✅ Models available: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'NOT FOUND'}")

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API is running"})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/overview")
def api_overview():
    results = []
    for ticker in US_STOCKS + INDIAN_STOCKS:
        try:
            r = predict(ticker, verbose=False)
            results.append(r)
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
