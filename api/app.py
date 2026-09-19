import sys, os
print("[BOOT] 1/8 starting imports", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required
from datetime import datetime, timedelta
print("[BOOT] 2/8 flask imports done, importing predict (torch etc)", flush=True)
from predict import predict, download_recent_data, compute_features
print("[BOOT] 3/8 predict module imported", flush=True)
from config import US_STOCKS, INDIAN_STOCKS, MODEL_DIR
import pandas as pd

from models import db, bcrypt
from auth import auth_bp
print("[BOOT] 4/8 all imports done, creating Flask app", flush=True)

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret-change-me")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)

database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(os.path.dirname(__file__), "app.db")
)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"connect_timeout": 10}}
print("[BOOT] 5/8 config set, DB target host:", database_url.split("@")[-1] if "@" in database_url else database_url, flush=True)

db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)
app.register_blueprint(auth_bp)
print("[BOOT] 6/8 extensions initialized, about to connect to DB", flush=True)

with app.app_context():
    db.create_all()
print("[BOOT] 7/8 db.create_all() succeeded", flush=True)

print("MODEL_DIR =", MODEL_DIR)
print("[BOOT] 8/8 startup complete, app ready", flush=True)

@app.route("/")
def home():
    return jsonify({"status": "ok"})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/predict")
@jwt_required()
def api_predict():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        return jsonify(predict(ticker, verbose=False))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chart")
@jwt_required()
def api_chart():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        from predict import NSE_SUFFIX, INDIAN_STOCKS, US_STOCKS
        bare = ticker.removesuffix(NSE_SUFFIX)
        is_indian = ticker in INDIAN_STOCKS or bare in INDIAN_STOCKS
        yf_ticker = bare + NSE_SUFFIX if is_indian else ticker
        raw = download_recent_data(yf_ticker)
        feat = compute_features(raw).tail(60).reset_index()
        date_col = "Date" if "Date" in feat.columns else feat.columns[0]
        def sl(c):
            return [None if pd.isna(x) else round(float(x),4) for x in feat[c]] if c in feat.columns else [None]*len(feat)
        return jsonify({
            "dates": [str(d)[:10] for d in feat[date_col]],
            "close": sl("Close"), "sma20": sl("SMA_20"),
            "bb_upper": sl("BB_Upper"), "bb_lower": sl("BB_Lower"),
            "rsi": sl("RSI"), "volume": sl("Volume")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/overview")
@jwt_required()
def api_overview():
    results = []
    for t in US_STOCKS + INDIAN_STOCKS:
        try:
            results.append(predict(t, verbose=False))
        except Exception as e:
            results.append({"ticker": t, "direction": "ERROR", "probability": 0, "proxy_used": t in INDIAN_STOCKS})
    return jsonify({"results": results, "total": len(results), "timestamp": datetime.now().isoformat()})

@app.route("/api/performance")
@jwt_required()
def api_performance():
    metrics = {
        "AAPL":  {"auc": 0.58, "accuracy": 0.56, "precision": 0.55, "recall": 0.57, "f1": 0.56},
        "MSFT":  {"auc": 0.61, "accuracy": 0.59, "precision": 0.58, "recall": 0.60, "f1": 0.59},
        "GOOGL": {"auc": 0.57, "accuracy": 0.55, "precision": 0.54, "recall": 0.56, "f1": 0.55},
        "AMZN":  {"auc": 0.59, "accuracy": 0.57, "precision": 0.56, "recall": 0.58, "f1": 0.57},
        "NVDA":  {"auc": 0.63, "accuracy": 0.60, "precision": 0.61, "recall": 0.59, "f1": 0.60},
    }
    return jsonify({"metrics": metrics})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
