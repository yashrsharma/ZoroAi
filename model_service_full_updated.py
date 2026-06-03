
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


def fetch_run_by_id(run_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    con = _connect(db_path)
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


class FinanceMLService:
    """
    Trains models and serves predictions.
    Stores prediction outputs in SQLite so dashboard can fetch latest/any run.
    """
    def __init__(self, dataset_path: Optional[str] = None, seed: int = 42):
        self.dataset_path = dataset_path or os.getenv("DATASET_PATH")  # set to your CSV path
        self.seed = seed

        self.df: Optional[pd.DataFrame] = None
        self.category_means: Dict[str, float] = {}

        self.debt_model: Optional[RandomForestRegressor] = None
        self.days_model: Optional[RandomForestRegressor] = None
        self.inv_model: Optional[RandomForestRegressor] = None

        self.metrics: Dict[str, Any] = {}

    def load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "DATASET_PATH not found. Set environment variable DATASET_PATH to your CSV path.\n"
                "Example (Windows PowerShell):  setx DATASET_PATH \"C:\\path\\to\\data.csv\"\n"
                "Then restart your terminal."
            )
        df = pd.read_csv(self.dataset_path)

        missing = [c for c in (["Income"] + EXPENSE_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Your CSV is missing required columns: {missing}")

        return df

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Targets and engineered fields (same logic you wrote, but safe)
        df = df.copy()

        df["Total_Expenses"] = df[EXPENSE_COLS].sum(axis=1)
        df["Actual_Savings"] = df["Income"] - df["Total_Expenses"]
        df["Savings_Rate"] = df["Actual_Savings"] / df["Income"].replace(0, np.nan)
        df["Savings_Rate"] = df["Savings_Rate"].fillna(0).clip(-5, 5)

        df["Debt_Score_Target"] = (df["Savings_Rate"] * 70 + (1 - (df["Loan_Repayment"]/df["Income"].replace(0, np.nan))) * 30)
        df["Debt_Score_Target"] = (df["Debt_Score_Target"] * 100).fillna(0).clip(0, 100)

        df["Days_To_Clear_Debt"] = (df["Loan_Repayment"] * 12 / (df["Actual_Savings"] + 1)).fillna(3650).clip(0, 3650)

        # synthetic investment used for training
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

        # Model 1: debt score
        y1 = df["Debt_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=self.seed)
        self.debt_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.debt_model.fit(X_train, y_train)
        debt_r2 = float(r2_score(y_test, self.debt_model.predict(X_test)))

        # Model 2: days predictor
        y2 = df["Days_To_Clear_Debt"]
        X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=self.seed)
        self.days_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.days_model.fit(X_train, y_train)
        days_r2 = float(r2_score(y_test, self.days_model.predict(X_test)))

        # Model 3: investment score predictor
        y3 = df["Investment_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=self.seed)
        self.inv_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.inv_model.fit(X_train, y_train)
        inv_r2 = float(r2_score(y_test, self.inv_model.predict(X_test)))

        self.metrics = {
            "debt_r2": debt_r2,
            "days_r2": days_r2,
            "investment_r2": inv_r2,
            "trained_rows": int(len(df)),
        }
        return self.metrics

    def _assert_trained(self) -> None:
        if self.debt_model is None or self.days_model is None or self.inv_model is None or not self.category_means:
            raise RuntimeError("Model not trained yet. Call /api/train (or start server which trains at boot).")

    def predict_and_store(
        self,
        income: float,
        rent: float,
        groceries: float,
        transport: float,
        entertainment: float,
        investment: float,
        timely_loan_repayment: float,
        goal_achievement: float,
        override_loan_repayment: Optional[float] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Runs prediction and stores output in SQLite.
        Returns (run_id, output_dict).
        """
        self._assert_trained()

        # Estimate other categories using dataset mean
        expenses = dict(self.category_means)

        expenses["Rent"] = float(rent)
        expenses["Groceries"] = float(groceries)
        expenses["Transport"] = float(transport)
        expenses["Entertainment"] = float(entertainment)

        if override_loan_repayment is not None:
            expenses["Loan_Repayment"] = float(override_loan_repayment)

        total_expenses = float(sum(expenses.values()))
        total_outflow = float(total_expenses + investment)

        actual_savings = float(income - total_outflow)
        savings_rate = float(actual_savings / income) if income and income > 0 else 0.0
        loan_repayment = float(expenses.get("Loan_Repayment", 0.0))

        input_data = pd.DataFrame([[income, total_expenses, actual_savings, savings_rate, loan_repayment, investment]],
                                  columns=FEATURES)

        predicted_debt_score = float(self.debt_model.predict(input_data)[0])
        predicted_days = float(self.days_model.predict(input_data)[0])
        predicted_investment_score = float(self.inv_model.predict(input_data)[0])

        finance_health_score = 0.6 * float(timely_loan_repayment) + 0.4 * float(goal_achievement)
        finance_health_score = float(round(finance_health_score, 2))

        upi_spend = float(expenses["Groceries"] + expenses["Transport"] + expenses.get("Eating_Out", 0) + expenses["Entertainment"] + expenses.get("Miscellaneous", 0))
        bills_spend = float(expenses["Rent"] + expenses.get("Utilities", 0) + expenses.get("Insurance", 0) + expenses.get("Healthcare", 0) + expenses.get("Education", 0))
        debt_spend = float(expenses.get("Loan_Repayment", 0))

        spend_groups = {
            "UPI Spend": round(upi_spend, 2),
            "Bills Spend": round(bills_spend, 2),
            "Debt Spend": round(debt_spend, 2),
            "Investment": round(float(investment), 2),
        }

        output = {
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
            "Grouped_Spending": spend_groups,
            "Model_Metrics": self.metrics
        }

        inputs = {
            "income": income,
            "rent": rent,
            "groceries": groceries,
            "transport": transport,
            "entertainment": entertainment,
            "investment": investment,
            "timely_loan_repayment": timely_loan_repayment,
            "goal_achievement": goal_achievement,
            "override_loan_repayment": override_loan_repayment
        }

        run_id = save_run(inputs, output, DB_PATH)
        return run_id, output
