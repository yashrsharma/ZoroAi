# storage.py
import sqlite3
import json
from datetime import datetime

DB_PATH = "finance_runs.db"

def init_db(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT NOT NULL,
            finance_health_score REAL,
            predicted_debt_score REAL,
            predicted_investment_score REAL
        )
    """)
    con.commit()
    con.close()

def save_run(inputs: dict, output: dict, db_path: str = DB_PATH) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    created_at = datetime.utcnow().isoformat()

    input_json = json.dumps(inputs, ensure_ascii=False)
    output_json = json.dumps(output, ensure_ascii=False)

    cur.execute("""
        INSERT INTO runs (
            created_at, input_json, output_json,
            finance_health_score, predicted_debt_score, predicted_investment_score
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        input_json,
        output_json,
        float(output.get("Finance_Health_Score", 0)),
        float(output.get("Predicted_Debt_Score", 0)),
        float(output.get("Predicted_Investment_Score", 0)),
    ))
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id

def fetch_run_by_id(run_id: int, db_path: str = DB_PATH) -> dict | None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, created_at, input_json, output_json FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "inputs": json.loads(row[2]),
        "output": json.loads(row[3]),
    }

def fetch_latest_run(db_path: str = DB_PATH) -> dict | None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, created_at, input_json, output_json FROM runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "inputs": json.loads(row[2]),
        "output": json.loads(row[3]),
    }

def fetch_best_by_finance_score(db_path: str = DB_PATH) -> dict | None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, input_json, output_json
        FROM runs
        ORDER BY finance_health_score DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "inputs": json.loads(row[2]),
        "output": json.loads(row[3]),
    }
