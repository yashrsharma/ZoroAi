from flask import Flask, jsonify, send_from_directory
import sqlite3
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "finance_runs.db")  # same DB your model script saves

def fetch_latest_run():
    if not os.path.exists(DB_PATH):
        return None, "Database not found. Run your model file once to create finance_runs.db"

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
        return None, "No runs saved yet. Run your model file once to insert output."

    run_id, created_at, input_json, output_json = row
    return {
        "id": run_id,
        "created_at": created_at,
        "inputs": json.loads(input_json),
        "output": json.loads(output_json)
    }, None

@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "Zoro API running", "db_path": DB_PATH})

@app.get("/api/latest")
def api_latest():
    run, err = fetch_latest_run()
    if err:
        return jsonify({"error": err}), 404
    return jsonify(run)

# OPTIONAL: serve dashboard from Flask so fetch works without CORS issues
@app.get("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
