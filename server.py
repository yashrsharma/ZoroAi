from flask import Flask, jsonify
import sqlite3
import json

app = Flask(__name__)
DB_PATH = "finance_runs.db"

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

@app.get("/api/latest")
def api_latest():
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT id, created_at, output_json
        FROM runs
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    con.close()

    if not row:
        return jsonify({"error": "No runs found in DB"}), 404

    return jsonify({
        "run_id": row["id"],
        "created_at": row["created_at"],
        "output": json.loads(row["output_json"])
    })

if __name__ == "__main__":
    # serve at http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
