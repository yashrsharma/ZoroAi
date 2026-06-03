import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DB_PATH = os.getenv("FINANCE_DB_PATH", "finance_runs.db")

EXPENSE_COLS = [
    'Rent','Loan_Repayment','Insurance','Groceries','Transport',
    'Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous'
]

FEATURES = ["Income","Total_Expenses","Actual_Savings","Savings_Rate","Loan_Repayment","Investment"]


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db(db_path: str = DB_PATH) -> None:
    con = _connect(db_path)
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


def save_run(inputs: Dict[str, Any], output: Dict[str, Any], db_path: str = DB_PATH) -> int:
    con = _connect(db_path)
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


def fetch_latest_run(db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    con = _connect(db_path)
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


class FinanceMLService:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.dataset_path = os.getenv("DATASET_PATH")

        self.df = None
        self.category_means: Dict[str, float] = {}

        self.debt_model = None
        self.days_model = None
        self.inv_model = None
        self.metrics: Dict[str, Any] = {}

    def load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "DATASET_PATH not found. Set environment variable DATASET_PATH to your CSV path.\n"
                "PowerShell (current session):  $env:DATASET_PATH = 'C:\\path\\to\\data.csv'\n"
            )

        df = pd.read_csv(self.dataset_path)

        missing = [c for c in (["Income"] + EXPENSE_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        return df

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Total_Expenses"] = df[EXPENSE_COLS].sum(axis=1)
        df["Actual_Savings"] = df["Income"] - df["Total_Expenses"]
        df["Savings_Rate"] = df["Actual_Savings"] / df["Income"].replace(0, np.nan)
        df["Savings_Rate"] = df["Savings_Rate"].fillna(0)

        df["Debt_Score_Target"] = (
            df["Savings_Rate"] * 70
            + (1 - (df["Loan_Repayment"] / df["Income"].replace(0, np.nan))) * 30
        )
        df["Debt_Score_Target"] = (df["Debt_Score_Target"] * 100).fillna(0).clip(0, 100)

        df["Days_To_Clear_Debt"] = (df["Loan_Repayment"] * 12 / (df["Actual_Savings"] + 1)).fillna(3650).clip(0, 3650)

        rng = np.random.default_rng(self.seed)
        df["Investment"] = (df["Actual_Savings"] * rng.uniform(0.2, 0.6, len(df))).clip(0)
        df["Investment_Rate"] = df["Investment"] / df["Income"].replace(0, np.nan)
        df["Investment_Score_Target"] = ((df["Investment_Rate"] / 0.20) * 100).fillna(0).clip(0, 100)

        return df

    def train(self) -> Dict[str, Any]:
        df_raw = self.load_dataset()
        df = self._prepare_training_frame(df_raw)

        self.df = df
        self.category_means = df[EXPENSE_COLS].mean(numeric_only=True).to_dict()

        X = df[FEATURES]

        y1 = df["Debt_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=self.seed)
        self.debt_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.debt_model.fit(X_train, y_train)
        debt_r2 = float(r2_score(y_test, self.debt_model.predict(X_test)))

        y2 = df["Days_To_Clear_Debt"]
        X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=self.seed)
        self.days_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.days_model.fit(X_train, y_train)
        days_r2 = float(r2_score(y_test, self.days_model.predict(X_test)))

        y3 = df["Investment_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=self.seed)
        self.inv_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.inv_model.fit(X_train, y_train)
        inv_r2 = float(r2_score(y_test, self.inv_model.predict(X_test)))

        self.metrics = {
            "debt_r2": debt_r2,
            "days_r2": days_r2,
            "investment_r2": inv_r2,
            "trained_rows": int(len(df))
        }
        return self.metrics

    def _assert_trained(self):
        if self.debt_model is None or self.days_model is None or self.inv_model is None:
            raise RuntimeError("Model not trained. Call /api/train first.")

    def predict_and_store(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        self._assert_trained()

        income = float(payload["income"])
        rent = float(payload["rent"])
        groceries = float(payload["groceries"])
        transport = float(payload["transport"])
        entertainment = float(payload["entertainment"])
        investment = float(payload["investment"])
        timely = float(payload["timely_loan_repayment"])
        goal = float(payload["goal_achievement"])

        expenses = dict(self.category_means)
        expenses["Rent"] = rent
        expenses["Groceries"] = groceries
        expenses["Transport"] = transport
        expenses["Entertainment"] = entertainment

        if "loan_repayment" in payload and payload["loan_repayment"] not in (None, ""):
            expenses["Loan_Repayment"] = float(payload["loan_repayment"])

        total_expenses = float(sum(expenses.values()))
        total_outflow = float(total_expenses + investment)
        actual_savings = float(income - total_outflow)
        savings_rate = float(actual_savings / income) if income > 0 else 0.0
        loan_repayment = float(expenses.get("Loan_Repayment", 0.0))

        X_in = pd.DataFrame([[income, total_expenses, actual_savings, savings_rate, loan_repayment, investment]], columns=FEATURES)

        predicted_debt_score = float(self.debt_model.predict(X_in)[0])
        predicted_days = float(self.days_model.predict(X_in)[0])
        predicted_investment_score = float(self.inv_model.predict(X_in)[0])

        finance_health_score = round(0.6 * timely + 0.4 * goal, 2)

        upi_spend = expenses["Groceries"] + expenses["Transport"] + expenses.get("Eating_Out", 0) + expenses["Entertainment"] + expenses.get("Miscellaneous", 0)
        bills_spend = expenses["Rent"] + expenses.get("Utilities", 0) + expenses.get("Insurance", 0) + expenses.get("Healthcare", 0) + expenses.get("Education", 0)
        debt_spend = expenses.get("Loan_Repayment", 0)

        output = {
            "Income": income,
            "Total_Expenses": round(total_expenses, 2),
            "Investment": investment,
            "Total_Outflow": round(total_outflow, 2),
            "Actual_Savings": round(actual_savings, 2),
            "Savings_Rate": round(savings_rate, 3),
            "Predicted_Debt_Score": round(predicted_debt_score, 2),
            "Predicted_Days_To_Clear_Debt": round(predicted_days, 2),
            "Predicted_Investment_Score": round(predicted_investment_score, 2),
            "Finance_Health_Score": finance_health_score,
            "Grouped_Spending": {
                "UPI Spend": round(float(upi_spend), 2),
                "Bills Spend": round(float(bills_spend), 2),
                "Debt Spend": round(float(debt_spend), 2),
                "Investment": round(float(investment), 2)
            },
            "Model_Metrics": self.metrics
        }

        run_id = save_run(payload, output, DB_PATH)
        return run_id, output
