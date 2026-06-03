import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ------------------------------
# LOAD DATASET
# ------------------------------
df = pd.read_csv(r"C:\Users\ARYAN AND ASTHA\Downloads\data (1) (1).csv")

expense_cols = [
    'Rent','Loan_Repayment','Insurance','Groceries','Transport',
    'Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous'
]

# ------------------------------
# 1️⃣ CREATE TARGETS FOR ML
# ------------------------------
df["Total_Expenses"] = df[expense_cols].sum(axis=1)
df["Actual_Savings"] = df["Income"] - df["Total_Expenses"]
df["Savings_Rate"] = df["Actual_Savings"] / df["Income"]

# ✅ Debt Score Target
df["Debt_Score_Target"] = (df["Savings_Rate"] * 70 + (1 - (df["Loan_Repayment"]/df["Income"])) * 30)
df["Debt_Score_Target"] = (df["Debt_Score_Target"] * 100).clip(0,100)

# ✅ Days to clear debt
df["Days_To_Clear_Debt"] = (df["Loan_Repayment"] * 12 / (df["Actual_Savings"]+1)).clip(0,3650)

# ✅ Add synthetic Investment Column for training
df["Investment"] = (df["Actual_Savings"] * np.random.uniform(0.2, 0.6, len(df))).clip(0)
df["Investment_Rate"] = df["Investment"] / df["Income"]

# ✅ Investment Score Target
df["Investment_Score_Target"] = (df["Investment_Rate"] / 0.20) * 100
df["Investment_Score_Target"] = df["Investment_Score_Target"].clip(0,100)

# ------------------------------
# 2️⃣ TRAIN ML MODELS
# ------------------------------
features = ["Income","Total_Expenses","Actual_Savings","Savings_Rate","Loan_Repayment","Investment"]
X = df[features]

# ✅ Model 1: Debt Score predictor
y1 = df["Debt_Score_Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)

debt_model = RandomForestRegressor(n_estimators=200, random_state=42)
debt_model.fit(X_train, y_train)
print("✅ Debt Score Model R2:", r2_score(y_test, debt_model.predict(X_test)))

# ✅ Model 2: Days predictor
y2 = df["Days_To_Clear_Debt"]
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)

days_model = RandomForestRegressor(n_estimators=200, random_state=42)
days_model.fit(X_train, y_train)
print("✅ Days Prediction Model R2:", r2_score(y_test, days_model.predict(X_test)))

# ✅ Model 3: Investment Score predictor
y3 = df["Investment_Score_Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=42)

inv_model = RandomForestRegressor(n_estimators=200, random_state=42)
inv_model.fit(X_train, y_train)
print("✅ Investment Score Model R2:", r2_score(y_test, inv_model.predict(X_test)))

# ------------------------------
# 3️⃣ USER FUNCTION (8 INPUTS)
# ------------------------------
category_means = df[expense_cols].mean().to_dict()

def finance_ml_model(income, rent, groceries, transport, entertainment, investment,
                     timely_loan_repayment, goal_achievement, show_charts=True):

    # Estimate other categories using dataset mean
    expenses = category_means.copy()
    expenses["Rent"] = rent
    expenses["Groceries"] = groceries
    expenses["Transport"] = transport
    expenses["Entertainment"] = entertainment

    # ✅ Expenses and Outflow
    total_expenses = sum(expenses.values())
    total_outflow = total_expenses + investment

    actual_savings = income - total_outflow
    savings_rate = actual_savings / income if income > 0 else 0
    loan_repayment = expenses["Loan_Repayment"]

    # ✅ ML Predictions
    input_data = pd.DataFrame([[income,total_expenses,actual_savings,savings_rate,loan_repayment,investment]],
                              columns=features)

    predicted_debt_score = debt_model.predict(input_data)[0]
    predicted_days = days_model.predict(input_data)[0]
    predicted_investment_score = inv_model.predict(input_data)[0]

    # ✅ PAYMENT GROUPING
    upi_spend = expenses["Groceries"] + expenses["Transport"] + expenses["Eating_Out"] + expenses["Entertainment"] + expenses["Miscellaneous"]
    bills_spend = expenses["Rent"] + expenses["Utilities"] + expenses["Insurance"] + expenses["Healthcare"] + expenses["Education"]
    debt_spend = expenses["Loan_Repayment"]

    spend_groups = {
        "UPI Spend": upi_spend,
        "Bills Spend": bills_spend,
        "Debt Spend": debt_spend,
        "Investment": investment
    }

    # ------------------------------
    # ✅ UPDATED DYNAMIC FINANCE HEALTH SCORE
    # ------------------------------

    # 1) Savings Score (ideal savings_rate = 0.30)
    savings_score = min((savings_rate / 0.30) * 100, 100)

    # 2) Debt Burden Score (loan repayment ideal <= 20% income)
    debt_ratio = loan_repayment / income if income > 0 else 1
    debt_score = max(0, 100 - (debt_ratio / 0.20) * 100)
    debt_score = min(debt_score, 100)

    # 3) Investment Score (ideal investment_rate >= 20%)
    investment_rate = investment / income if income > 0 else 0
    investment_score = min((investment_rate / 0.20) * 100, 100)

    # 4) Expense Discipline Score (essential expenses <= 50% income)
    essential_expenses = rent + groceries + expenses["Utilities"] + expenses["Insurance"]
    essential_ratio = essential_expenses / income if income > 0 else 1
    discipline_score = max(0, 100 - (essential_ratio / 0.50) * 100)
    discipline_score = min(discipline_score, 100)

    # 5) Behavior Score (your original inputs)
    behavior_score = 0.6 * timely_loan_repayment + 0.4 * goal_achievement

    # ✅ Final Finance Health Score
    finance_health_score = (
        0.30 * savings_score +
        0.20 * debt_score +
        0.20 * investment_score +
        0.15 * discipline_score +
        0.15 * behavior_score
    )
    finance_health_score = round(finance_health_score, 2)

    # ✅ Charts
    if show_charts:

        # Pie Chart
        plt.figure(figsize=(6,6))
        plt.pie(spend_groups.values(), labels=spend_groups.keys(), autopct="%1.1f%%")
        plt.title("Spending Breakdown (UPI vs Bills vs Debt vs Investment)")
        plt.show()

        months = list(range(1,13))

        # Savings projection
        savings_projection = [actual_savings * m for m in months]
        plt.figure(figsize=(8,4))
        plt.plot(months, savings_projection, label="Savings Growth")
        plt.title("Savings Projection (12 months)")
        plt.xlabel("Months")
        plt.ylabel("Amount")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Investment Score projection
        plt.figure(figsize=(6,4))
        plt.plot(months, [predicted_investment_score]*12)
        plt.title("Investment Score Projection")
        plt.xlabel("Months")
        plt.ylabel("Investment Score")
        plt.ylim(0,100)
        plt.grid(True)
        plt.show()

        # Finance Health Score projection
        plt.figure(figsize=(6,4))
        plt.plot(months, [finance_health_score]*12)
        plt.title("Finance Health Score Projection")
        plt.xlabel("Months")
        plt.ylabel("Finance Health Score")
        plt.ylim(0,100)
        plt.grid(True)
        plt.show()

    return {
        "Income": income,
        "Total_Expenses": round(total_expenses,2),
        "Investment": investment,
        "Total_Outflow": round(total_outflow,2),
        "Actual_Savings": round(actual_savings,2),
        "Savings_Rate": round(savings_rate,3),
        "Predicted_Debt_Score": round(predicted_debt_score,2),
        "Predicted_Days_To_Clear_Debt": round(predicted_days,2),
        "Predicted_Investment_Score": round(predicted_investment_score,2),
        "Finance_Health_Score": finance_health_score,
        "Finance_Health_Breakdown": {
            "Savings_Score": round(savings_score,2),
            "Debt_Score": round(debt_score,2),
            "Investment_Score": round(investment_score,2),
            "Discipline_Score": round(discipline_score,2),
            "Behavior_Score": round(behavior_score,2)
        },
        "Grouped_Spending": {k: round(v,2) for k,v in spend_groups.items()}
    }

# ------------------------------
# ✅ RUN EXAMPLE
# ------------------------------
output = finance_ml_model(
    income=200000,
    rent=12000,
    groceries=6000,
    transport=2000,
    entertainment=2000,
    investment=15000,
    timely_loan_repayment=90,
    goal_achievement=75,
    show_charts=True
)

print("\n✅ FINAL OUTPUT:\n")
for k,v in output.items():
    print(k, ":", v)