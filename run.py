import sqlite3
import json

DB_PATH = "finance_runs.db"


def show_last_runs(limit=10):
    """Print last N runs (id, created_at, finance_health_score)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT id, created_at, finance_health_score
        FROM runs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    con.close()

    print(rows)
    return rows


def get_latest_output():
    """Fetch latest run output_json as Python dict."""
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
        print("❌ No runs found in database.")
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

    print("\n✅ LATEST RUN")
    print("Run ID:", run_id)
    print("Created At:", created_at)
    print("\nINPUTS:")
    print(inputs)
    print("\nOUTPUT:")
    print(output)

    return {"run_id": run_id, "created_at": created_at, "inputs": inputs, "output": output}


if __name__ == "__main__":
    show_last_runs(limit=10)
    # Uncomment if you also want full latest output printed:
    # get_latest_output()
