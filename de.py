import sqlite3
import pandas as pd

# Path to your DB
DB_PATH = "finance_runs.db"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

# Load table into DataFrame
df = pd.read_sql_query("SELECT * FROM runs", conn)

# Close connection
conn.close()

# Save to CSV
df.to_csv("finance_runs.csv", index=False)

print("✅ finance_runs.csv created successfully")
