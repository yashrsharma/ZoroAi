"""
model.py
--------
Finance model + scoring utilities used by the Flask dashboard.

What you get:
- A trainable ML pipeline (RandomForest) for:
  * Debt score (0-100)
  * Days to clear debt (0-3650)
  * Investment score (0-100)
- A Finance Health Score (0-100) computed from timely_loan_repayment + goal_achievement
- Grouped spending buckets used for the dashboard ring graph.

This module is designed to be imported by app.py (server) and called via /api/* endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


EXPENSE_COLS = [
    "Rent",
    "Loan_Repayment",
    "Insurance",
    "Groceries",
    "Transport",
    "Eating_Out",
    "Entertainment",
    "Utilities",
    "Healthcare",
    "Education",
    "Miscellaneous",
]

FEATURES = ["Income", "Total_Expenses", "Actual_Savings", "Savings_Rate", "Loan_Repayment", "Investment"]


def _health_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Needs Attention"


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _generate_synthetic_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Fallback dataset so the app runs even if data.csv is not present.
    """
    rng = np.random.default_rng(seed)

    income = rng.normal(120_000, 45_000, n).clip(20_000, 400_000)
    rent = (income * rng.uniform(0.08, 0.22, n)).clip(4_000, 80_000)
    loan = (income * rng.uniform(0.00, 0.15, n)).clip(0, 80_000)
    insurance = rng.normal(2_000, 700, n).clip(0, 12_000)
    groceries = (income * rng.uniform(0.03, 0.10, n)).clip(1_000, 35_000)
    transport = rng.normal(3_500, 1_800, n).clip(0, 25_000)
    eating_out = rng.normal(4_500, 2_400, n).clip(0, 30_000)
    entertainment = rng.normal(3_200, 2_000, n).clip(0, 25_000)
    utilities = rng.normal(3_000, 1_200, n).clip(500, 15_000)
    healthcare = rng.normal(2_200, 1_500, n).clip(0, 25_000)
    education = rng.normal(1_500, 2_200, n).clip(0, 30_000)
    misc = rng.normal(2_800, 2_000, n).clip(0, 35_000)

    df = pd.DataFrame(
        {
            "Income": income,
            "Rent": rent,
            "Loan_Repayment": loan,
            "Insurance": insurance,
            "Groceries": groceries,
            "Transport": transport,
            "Eating_Out": eating_out,
            "Entertainment": entertainment,
            "Utilities": utilities,
            "Healthcare": healthcare,
            "Education": education,
            "Miscellaneous": misc,
        }
    )

    # Create some realistic structure: higher income tends to have higher spending
    for col in EXPENSE_COLS:
        if col != "Loan_Repayment":
            df[col] = (df[col] * rng.uniform(0.95, 1.10, n)).clip(0)

    return df


@dataclass
class TrainReport:
    debt_r2: float
    days_r2: float
    inv_r2: float
    rows: int
    dataset_path: str


class FinanceModel:
    def __init__(self, dataset_path: Optional[str] = None, seed: int = 42) -> None:
        self.seed = seed
        self.dataset_path = dataset_path
        self.df: Optional[pd.DataFrame] = None

        self.debt_model: Optional[RandomForestRegressor] = None
        self.days_model: Optional[RandomForestRegressor] = None
        self.inv_model: Optional[RandomForestRegressor] = None

        self.category_means: Dict[str, float] = {}

        self.last_train_report: Optional[TrainReport] = None

    def load_dataset(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        path = dataset_path or self.dataset_path
        if path:
            p = Path(path)
            if p.exists() and p.is_file():
                df = pd.read_csv(p)
            else:
                df = _generate_synthetic_dataset()
                p = Path("synthetic://generated")
        else:
            df = _generate_synthetic_dataset()
            p = Path("synthetic://generated")

        # Ensure required columns exist
        missing = [c for c in (["Income"] + EXPENSE_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Keep only needed cols and coerce numeric
        df = df[["Income"] + EXPENSE_COLS].copy()
        for c in ["Income"] + EXPENSE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        self.df = df
        self.dataset_path = str(p)
        self.category_means = df[EXPENSE_COLS].mean(numeric_only=True).to_dict()
        return df

    def train(self, dataset_path: Optional[str] = None) -> TrainReport:
        df = self.load_dataset(dataset_path)

        # Targets
        df["Total_Expenses"] = df[EXPENSE_COLS].sum(axis=1)
        df["Actual_Savings"] = df["Income"] - df["Total_Expenses"]
        df["Savings_Rate"] = df["Actual_Savings"] / df["Income"]

        df["Debt_Score_Target"] = (df["Savings_Rate"] * 0.70 + (1 - (df["Loan_Repayment"] / df["Income"])) * 0.30)
        df["Debt_Score_Target"] = (df["Debt_Score_Target"] * 100).clip(0, 100)

        df["Days_To_Clear_Debt"] = (df["Loan_Repayment"] * 12 / (df["Actual_Savings"] + 1)).clip(0, 3650)

        rng = np.random.default_rng(self.seed)
        df["Investment"] = (df["Actual_Savings"] * rng.uniform(0.2, 0.6, len(df))).clip(0)
        df["Investment_Rate"] = df["Investment"] / df["Income"]

        df["Investment_Score_Target"] = ((df["Investment_Rate"] / 0.20) * 100).clip(0, 100)

        X = df[FEATURES].copy()

        # Debt model
        y1 = df["Debt_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=self.seed)
        self.debt_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.debt_model.fit(X_train, y_train)
        debt_r2 = float(r2_score(y_test, self.debt_model.predict(X_test)))

        # Days model
        y2 = df["Days_To_Clear_Debt"]
        X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=self.seed)
        self.days_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.days_model.fit(X_train, y_train)
        days_r2 = float(r2_score(y_test, self.days_model.predict(X_test)))

        # Investment model
        y3 = df["Investment_Score_Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=self.seed)
        self.inv_model = RandomForestRegressor(n_estimators=200, random_state=self.seed)
        self.inv_model.fit(X_train, y_train)
        inv_r2 = float(r2_score(y_test, self.inv_model.predict(X_test)))

        report = TrainReport(
            debt_r2=debt_r2,
            days_r2=days_r2,
            inv_r2=inv_r2,
            rows=int(len(df)),
            dataset_path=str(self.dataset_path or ""),
        )
        self.last_train_report = report
        return report

    def predict(self, *, income: float, rent: float, groceries: float, transport: float, entertainment: float,
                investment: float, timely_loan_repayment: float, goal_achievement: float) -> Dict[str, Any]:

        if not (self.debt_model and self.days_model and self.inv_model):
            raise RuntimeError("Models not trained. Call train() first.")

        # Estimate non-provided categories using mean values
        expenses = dict(self.category_means) if self.category_means else {k: 0.0 for k in EXPENSE_COLS}
        expenses["Rent"] = float(rent)
        expenses["Groceries"] = float(groceries)
        expenses["Transport"] = float(transport)
        expenses["Entertainment"] = float(entertainment)

        total_expenses = float(sum(expenses.values()))
        total_outflow = total_expenses + float(investment)

        actual_savings = float(income) - total_outflow
        savings_rate = (actual_savings / float(income)) if income > 0 else 0.0
        loan_repayment = float(expenses.get("Loan_Repayment", 0.0))

        input_df = pd.DataFrame(
            [[income, total_expenses, actual_savings, savings_rate, loan_repayment, investment]],
            columns=FEATURES,
        )

        predicted_debt_score = float(self.debt_model.predict(input_df)[0])
        predicted_days = float(self.days_model.predict(input_df)[0])
        predicted_investment_score = float(self.inv_model.predict(input_df)[0])

        # Health score (0-100) from user-behavior signals
        timely_loan_repayment = float(timely_loan_repayment)
        goal_achievement = float(goal_achievement)

        # Normalize to 0..100 if user gives 0..1
        if timely_loan_repayment <= 1.0 and goal_achievement <= 1.0:
            timely_loan_repayment *= 100.0
            goal_achievement *= 100.0

        finance_health_score = (0.6 * timely_loan_repayment + 0.4 * goal_achievement)
        finance_health_score = float(np.clip(finance_health_score, 0, 100))
        health_label = _health_label(finance_health_score)

        # Grouped spending for ring chart
        upi_spend = float(expenses.get("Groceries", 0.0) + expenses.get("Transport", 0.0) +
                          expenses.get("Eating_Out", 0.0) + expenses.get("Entertainment", 0.0) +
                          expenses.get("Miscellaneous", 0.0))
        bills_spend = float(expenses.get("Rent", 0.0) + expenses.get("Utilities", 0.0) +
                            expenses.get("Insurance", 0.0) + expenses.get("Healthcare", 0.0) +
                            expenses.get("Education", 0.0))
        debt_spend = float(expenses.get("Loan_Repayment", 0.0))

        spend_groups = {
            "UPI Spend": round(upi_spend, 2),
            "Bills Spend": round(bills_spend, 2),
            "Debt Spend": round(debt_spend, 2),
            "Investment": round(float(investment), 2),
        }

        total_for_ring = sum(spend_groups.values()) or 1.0
        ring_percent = {k: round((v / total_for_ring) * 100, 2) for k, v in spend_groups.items()}

        return {
            "Income": float(income),
            "Total_Expenses": round(total_expenses, 2),
            "Investment": round(float(investment), 2),
            "Total_Outflow": round(total_outflow, 2),
            "Actual_Savings": round(actual_savings, 2),
            "Savings_Rate": round(float(_clip01(savings_rate)), 4),
            "Predicted_Debt_Score": round(predicted_debt_score, 2),
            "Predicted_Days_To_Clear_Debt": round(predicted_days, 2),
            "Predicted_Investment_Score": round(predicted_investment_score, 2),
            "Finance_Health_Score": round(finance_health_score, 2),
            "Finance_Health_Label": health_label,
            "Grouped_Spending": spend_groups,
            "Ring_Percent": ring_percent,
            "Train_Report": (self.last_train_report.__dict__ if self.last_train_report else None),
        }


def default_demo_inputs() -> Dict[str, float]:
    """
    Used by the dashboard to show data without needing a form.
    (You can wire this up to real user inputs later.)
    """
    return dict(
        income=200000,
        rent=12000,
        groceries=6000,
        transport=2000,
        entertainment=2000,
        investment=15000,
        timely_loan_repayment=90,
        goal_achievement=75,
    )
