# ============================================================
# ✅ FULLY UPDATED CODE (MATCHES YOUR UPLOADED FILE data (1).csv)
# - Uses the exact expense columns that exist in your CSV
# - Fixes DB schema mismatch automatically (adds missing columns)
# - Fixes utcnow() warning (uses timezone-aware UTC)
# - Makes user calculations based ONLY on user inputs (no hidden means)
# - Still trains ML models using the dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ==============================
# CONFIG
# ==============================
# ✅ Set this to your PC path:
CSV_PATH = r"C:\Users\YASH SHARMA\Downloads\data (1).csv"
# (In this chat sandbox, your file is at: /mnt/data/data (1).csv)

DB_PATH = "finance_runs.db"

# These are the expense columns present in your file
EXPENSE_COLS = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"
]


# ==============================
# DATABASE (SQLite)
# ==============================
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Create if not exists (latest schema)
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

    # Auto-migrate older tables by adding missing columns
    cur.execute("PRAGMA table_info(runs)")
    existing_cols = {row[1] for row in cur.fetchall()}

    needed = {
        "finance_health_score": "REAL",
        "predicted_debt_score": "REAL",
        "predicted_investment_score": "REAL",
        "predicted_days_to_clear_debt": "REAL",
    }
    for col, coltype in needed.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE runs ADD COLUMN {col} {coltype}")

    con.commit()
    con.close()


def save_output_to_db(inputs: dict, output: dict) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    created_at = datetime.now(timezone.utc).isoformat()  # ✅ timezone-aware UTC
    input_json = json.dumps(inputs, ensure_ascii=False)
    output_json = json.dumps(output, ensure_ascii=False)

    cur.execute("""
        INSERT INTO runs (
            created_at, input_json, output_json,
            finance_health_score, predicted_debt_score, predicted_investment_score, predicted_days_to_clear_debt
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        input_json,
        output_json,
        float(output.get("Finance_Health_Score", 0)),
        float(output.get("Predicted_Debt_Score", 0)),
        float(output.get("Predicted_Investment_Score", 0)),
        float(output.get("Predicted_Days_To_Clear_Debt", 0)),
    ))

    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


# ==============================
# HELPERS
# ==============================
def clamp(x, lo=0.0, hi=100.0):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(lo, min(hi, x))


# ==============================
# LOAD DATASET (YOUR FILE)
# ==============================
df = pd.read_csv(CSV_PATH)

# Validate required columns
missing = [c for c in (["Income"] + EXPENSE_COLS) if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# ==============================
# 1️⃣ CREATE CONSISTENT TRAINING FEATURES + TARGETS
# ==============================
df["Total_Expenses"] = df[EXPENSE_COLS].sum(axis=1)

# Your dataset doesn't contain "Investment", so create a deterministic one (NO randomness):
# investment = 20% of positive leftover after expenses
df["Investment"] = (0.20 * (df["Income"] - df["Total_Expenses"])).clip(lower=0)

df["Total_Outflow"] = df["Total_Expenses"] + df["Investment"]
df["Actual_Savings"] = df["Income"] - df["Total_Outflow"]
df["Savings_Rate"] = np.where(df["Income"] > 0, df["Actual_Savings"] / df["Income"], 0.0)

df["Debt_Ratio"] = np.where(df["Income"] > 0, df["Loan_Repayment"] / df["Income"], 1.0)
df["Investment_Rate"] = np.where(df["Income"] > 0, df["Investment"] / df["Income"], 0.0)

# Targets (0–100)
df["Debt_Score_Target"] = (100 - (df["Debt_Ratio"] / 0.40) * 100).clip(0, 100)
df["Investment_Score_Target"] = ((df["Investment_Rate"] / 0.20) * 100).clip(0, 100)

# Days-to-clear-debt target:
# If savings <= 0 -> 3650 (10 years), else approximate days using annual repayment burden / monthly savings
monthly_savings = df["Actual_Savings"].clip(lower=0)
df["Days_To_Clear_Debt"] = np.where(
    monthly_savings > 0,
    (df["Loan_Repayment"] * 12) / (monthly_savings + 1e-6) * 30,
    3650
).clip(0, 3650)

# ==============================
# 2️⃣ TRAIN ML MODELS
# ==============================
features = ["Income", "Total_Expenses", "Actual_Savings", "Savings_Rate", "Loan_Repayment", "Investment"]
X = df[features]

# Debt score model
y1 = df["Debt_Score_Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
debt_model = RandomForestRegressor(n_estimators=300, random_state=42)
debt_model.fit(X_train, y_train)
print("✅ Debt Score Model R2:", r2_score(y_test, debt_model.predict(X_test)))

# Days model
y2 = df["Days_To_Clear_Debt"]
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)
days_model = RandomForestRegressor(n_estimators=300, random_state=42)
days_model.fit(X_train, y_train)
print("✅ Days Prediction Model R2:", r2_score(y_test, days_model.predict(X_test)))

# Investment score model
y3 = df["Investment_Score_Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=42)
inv_model = RandomForestRegressor(n_estimators=300, random_state=42)
inv_model.fit(X_train, y_train)
print("✅ Investment Score Model R2:", r2_score(y_test, inv_model.predict(X_test)))


# ==============================
# 3️⃣ USER FUNCTION (CALCS BASED ONLY ON USER INPUTS)
# ==============================
def finance_ml_model(
    income,
    rent=0.0,
    loan_repayment=0.0,
    insurance=0.0,
    groceries=0.0,
    transport=0.0,
    eating_out=0.0,
    entertainment=0.0,
    utilities=0.0,
    healthcare=0.0,
    education=0.0,
    miscellaneous=0.0,
    investment=0.0,
    timely_loan_repayment=0.0,   # 0–100
    goal_achievement=0.0,        # 0–100
    show_charts=True
):
    """
    ✅ All financial calculations depend ONLY on the values you pass here.
    No hidden dataset means are added.

    Returns:
      - totals (expenses/outflow/savings)
      - ML predictions (based on trained models)
      - finance health score (rule-based, clamped 0–100)
      - spending breakdown pie chart (optional)
    """

    # Build expense dict from user inputs (matches your dataset columns)
    expenses = {
        "Rent": float(rent),
        "Loan_Repayment": float(loan_repayment),
        "Insurance": float(insurance),
        "Groceries": float(groceries),
        "Transport": float(transport),
        "Eating_Out": float(eating_out),
        "Entertainment": float(entertainment),
        "Utilities": float(utilities),
        "Healthcare": float(healthcare),
        "Education": float(education),
        "Miscellaneous": float(miscellaneous),
    }

    total_expenses = float(sum(expenses.values()))
    total_outflow = float(total_expenses + float(investment))

    actual_savings = float(income - total_outflow)
    savings_rate = (actual_savings / income) if income > 0 else 0.0

    # ML input
    input_data = pd.DataFrame([[
        float(income),
        float(total_expenses),
        float(actual_savings),
        float(savings_rate),
        float(loan_repayment),
        float(investment)
    ]], columns=features)

    predicted_debt_score = float(debt_model.predict(input_data)[0])
    predicted_days = float(days_model.predict(input_data)[0])
    predicted_investment_score = float(inv_model.predict(input_data)[0])

    # ==========================
    # Rule-based scoring (0–100)
    # ==========================
    # Savings score: ideal >= 30%
    savings_score = clamp((max(savings_rate, 0.0) / 0.30) * 100)

    # Debt score: ideal loan <= 20% income; 0 at 40%+
    debt_ratio = (loan_repayment / income) if income > 0 else 1.0
    debt_score = clamp(100 - (debt_ratio / 0.40) * 100)

    # Investment score: ideal >= 20%
    investment_rate = (investment / income) if income > 0 else 0.0
    investment_score = clamp((investment_rate / 0.20) * 100)

    # Discipline score: essential <= 50% income
    essential_expenses = rent + groceries + utilities + insurance
    essential_ratio = (essential_expenses / income) if income > 0 else 1.0
    discipline_score = clamp(100 - (essential_ratio / 0.80) * 100)  # 0 at 80%+

    # Behavior score
    behavior_score = clamp(0.6 * timely_loan_repayment + 0.4 * goal_achievement)

    finance_health_score = (
        0.30 * savings_score +
        0.20 * debt_score +
        0.20 * investment_score +
        0.15 * discipline_score +
        0.15 * behavior_score
    )
    finance_health_score = round(clamp(finance_health_score, 0, 100), 2)

    # Spending groups
    upi_spend = groceries + transport + eating_out + entertainment + miscellaneous
    bills_spend = rent + utilities + insurance + healthcare + education
    debt_spend = loan_repayment

    spend_groups = {
        "UPI Spend": float(upi_spend),
        "Bills Spend": float(bills_spend),
        "Debt Spend": float(debt_spend),
        "Investment": float(investment)
    }

    if show_charts:
        plt.figure(figsize=(6, 6))
        plt.pie(spend_groups.values(), labels=spend_groups.keys(), autopct="%1.1f%%")
        plt.title("Spending Breakdown (UPI vs Bills vs Debt vs Investment)")
        plt.show()

    return {
        "Income": float(income),
        "Total_Expenses": round(total_expenses, 2),
        "Investment": float(investment),
        "Total_Outflow": round(total_outflow, 2),
        "Actual_Savings": round(actual_savings, 2),
        "Savings_Rate": round(savings_rate, 3),

        "Predicted_Debt_Score": round(predicted_debt_score, 2),
        "Predicted_Days_To_Clear_Debt": round(predicted_days, 2),
        "Predicted_Investment_Score": round(predicted_investment_score, 2),

        "Finance_Health_Score": finance_health_score,
        "Grouped_Spending": {k: round(v, 2) for k, v in spend_groups.items()}
    }


# ==============================
# ✅ RUN EXAMPLE + SAVE TO DB
# ==============================
if __name__ == "__main__":
    init_db()

    inputs = dict(
        income=230000,
        rent=10000,
        groceries=6060,
        transport=2000,
        entertainment=5000,
        investment=45000,
        loan_repayment=10040,            # ✅ no semicolon
        timely_loan_repayment=90,
        goal_achievement=75,

        # Optional: if you want realistic totals, add these too:
        # utilities=800, insurance=500, eating_out=1000, miscellaneous=700, etc.
    )

    output = finance_ml_model(**inputs, show_charts=True)

    run_id = save_output_to_db(inputs, output)
    print("\n✅ Saved output to DB with run_id:", run_id)

    print("\n✅ FINAL OUTPUT:\n")
    for k, v in output.items():
        print(k, ":", v)
