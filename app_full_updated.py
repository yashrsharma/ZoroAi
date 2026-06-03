import os
import sqlite3
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# If you still want /api/predict to work via your service, keep this import.
# Otherwise you can remove it.
from model_service_full_updated import FinanceMLService

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ IMPORTANT: Use the SAME DB your model writes to
DB_PATH = os.path.join(BASE_DIR, "finance_runs.db")


def init_db_sqlite():
    """Ensure the runs table exists (matches your model_with_dp schema)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT NOT NULL,
            finance_health_score REAL,
            predicted_debt_score REAL,
            predicted_investment_score REAL,
            predicted_days_to_clear_debt REAL
        )
    """)
    con.commit()
    con.close()


def fetch_latest_run_sqlite():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, input_json, output_json
        FROM runs
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    run_id, created_at, input_json, output_json = row

    try:
        inputs = json.loads(input_json)
    except Exception:
        inputs = input_json

    try:
        output = json.loads(output_json)
    except Exception:
        output = output_json

    return {
        "run_id": run_id,
        "created_at": created_at,
        "inputs": inputs,
        "output": output
    }


def fetch_run_by_id_sqlite(run_id: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, input_json, output_json
        FROM runs
        WHERE id = ?
        LIMIT 1
    """, (run_id,))
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    rid, created_at, input_json, output_json = row

    try:
        inputs = json.loads(input_json)
    except Exception:
        inputs = input_json

    try:
        output = json.loads(output_json)
    except Exception:
        output = output_json

    return {
        "run_id": rid,
        "created_at": created_at,
        "inputs": inputs,
        "output": output
    }


# Init DB once
init_db_sqlite()

service = FinanceMLService()

@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "Finance ML API running", "db_path": DB_PATH})


@app.get("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard_updated.html")


@app.get("/api/dbinfo")
def dbinfo():
    return jsonify({
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR,
        "db_path": DB_PATH,
        "db_exists": os.path.exists(DB_PATH),
        "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else None,
        "base_dir_files_sample": os.listdir(BASE_DIR)[:50],
    })


@app.get("/api/latest")
def api_latest():
    latest = fetch_latest_run_sqlite()
    if not latest:
        return jsonify({"error": "No runs saved yet in finance_runs.db"}), 404
    return jsonify(latest)


@app.get("/api/run/<int:run_id>")
def api_run(run_id: int):
    run = fetch_run_by_id_sqlite(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(run)


# OPTIONAL: keep predict endpoint (stores via your service)
@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True) or {}

    required = ["income","rent","groceries","transport","entertainment","investment","timely_loan_repayment","goal_achievement"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # NOTE: Make sure your FinanceMLService writes into finance_runs.db too.
        run_id, out = service.predict_and_store(
            income=float(payload["income"]),
            rent=float(payload["rent"]),
            groceries=float(payload["groceries"]),
            transport=float(payload["transport"]),
            entertainment=float(payload["entertainment"]),
            investment=float(payload["investment"]),
            timely_loan_repayment=float(payload["timely_loan_repayment"]),
            goal_achievement=float(payload["goal_achievement"]),
            override_loan_repayment=float(payload["loan_repayment"]) if "loan_repayment" in payload else None
        )
        return jsonify({"run_id": run_id, "output": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
