from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory, make_response

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_updated import FinanceModel, default_demo_inputs

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

# ----------------------------
# Model bootstrap (train once)
# ----------------------------
DATASET_PATH = os.environ.get("DATASET_PATH")  # optionally set to a CSV path
finance_model = FinanceModel(dataset_path=DATASET_PATH)

# Train at startup so dashboard is always "updated" with the latest trained model
try:
    finance_model.train()
except Exception as e:
    # If training fails, still start server; API will show the error message
    finance_model.last_train_report = None
    print("⚠️ Model training failed:", e)


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ----------------------------
# Pages
# ----------------------------
@app.get("/")
def landing():
    return _no_cache(send_from_directory(BASE_DIR, "zoro.html"))


@app.get("/login")
def login():
    return _no_cache(send_from_directory(BASE_DIR, "index.html"))


@app.get("/dashboard")
def dashboard():
    # Use the updated dashboard file
    return _no_cache(send_from_directory(BASE_DIR, "dashboard_updated.html"))


# Keep old route working if someone browses directly
@app.get("/dashboard.html")
def dashboard_html():
    return _no_cache(send_from_directory(BASE_DIR, "dashboard_updated.html"))


# ----------------------------
# APIs
# ----------------------------
@app.get("/api/metrics")
def api_metrics():
    """
    Returns the latest computed metrics for a demo user input.
    You can override inputs via query params (e.g. ?income=250000&rent=15000).
    """
    payload: Dict[str, Any] = default_demo_inputs()

    # Allow overriding defaults via query params
    for k in list(payload.keys()):
        if k in request.args:
            try:
                payload[k] = float(request.args.get(k, payload[k]))
            except ValueError:
                pass

    try:
        out = finance_model.predict(**payload)
        resp = make_response(jsonify(out), 200)
    except Exception as e:
        resp = make_response(jsonify({"error": str(e)}), 500)

    return _no_cache(resp)


@app.post("/api/predict")
def api_predict():
    """
    POST JSON with:
      income, rent, groceries, transport, entertainment, investment,
      timely_loan_repayment, goal_achievement
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        out = finance_model.predict(
            income=float(data.get("income", 0)),
            rent=float(data.get("rent", 0)),
            groceries=float(data.get("groceries", 0)),
            transport=float(data.get("transport", 0)),
            entertainment=float(data.get("entertainment", 0)),
            investment=float(data.get("investment", 0)),
            timely_loan_repayment=float(data.get("timely_loan_repayment", 0)),
            goal_achievement=float(data.get("goal_achievement", 0)),
        )
        resp = make_response(jsonify(out), 200)
    except Exception as e:
        resp = make_response(jsonify({"error": str(e)}), 400)
    return _no_cache(resp)


@app.post("/api/train")
def api_train():
    """
    Retrain the model.
    Options:
    - Provide JSON { "dataset_path": "path/to/data.csv" }
    - Or set DATASET_PATH env var and call without body.

    Returns training report.
    """
    data = request.get_json(force=True, silent=True) or {}
    dataset_path = data.get("dataset_path") or finance_model.dataset_path
    try:
        report = finance_model.train(dataset_path)
        resp = make_response(jsonify({"ok": True, "train_report": report.__dict__}), 200)
    except Exception as e:
        resp = make_response(jsonify({"ok": False, "error": str(e)}), 500)
    return _no_cache(resp)


if __name__ == "__main__":
    app.run(debug=True)
