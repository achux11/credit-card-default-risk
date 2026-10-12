"""
Credit Card Default Prediction - starter project
Dataset: "Default of Credit Card Clients" (UCI / Kaggle, 30,000 clients, Taiwan)
  Save the Kaggle CSV next to this script as: UCI_Credit_Card.csv

Run:  pip install pandas scikit-learn xgboost matplotlib
      python credit_card_default_project.py

Outputs (in ./outputs):
  metrics.txt, confusion_matrix.png, roc_curve.png,
  scored_customers.csv  (load this into Power BI for the dashboard)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)
RANDOM_STATE = 42

# ---------- 1. Load and clean ----------
df = pd.read_csv("UCI_Credit_Card.csv")
df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
df = df.drop(columns=["ID"])
# Undocumented codes -> "other"
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# ---------- 2. Feature engineering (behavioral credit features) ----------
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
paid = [f"PAY_AMT{i}" for i in range(1, 7)]
status = [f"PAY_{i}" for i in range(1, 7)]
lim = df["LIMIT_BAL"].replace(0, np.nan)

df["utilization_latest"] = df["BILL_AMT1"] / lim
df["utilization_avg_6m"] = df[bill].mean(axis=1) / lim
df["pay_ratio_avg_6m"] = (df[paid].sum(axis=1) / df[bill].clip(lower=1).sum(axis=1)).clip(upper=5)
df["months_delinquent"] = (df[status] > 0).sum(axis=1)
df["max_delinquency"] = df[status].max(axis=1)
df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
df = df.fillna(0)

X = df.drop(columns=["default"])
y = df["default"]

# ---------- 3. Split: train / validation / test (70/15/15, stratified) ----------
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

# ---------- 4. Model with early stopping on validation ----------
model = XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, eval_metric="aucpr", early_stopping_rounds=100,
    random_state=RANDOM_STATE,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

# ---------- 5. Pick threshold on VALIDATION only (max F1) ----------
p_va = model.predict_proba(X_va)[:, 1]
grid = np.arange(0.05, 0.95, 0.01)
best_t = grid[np.argmax([f1_score(y_va, p_va >= t) for t in grid])]

# ---------- 6. Evaluate once on locked TEST ----------
p_te = model.predict_proba(X_te)[:, 1]
pred_te = (p_te >= best_t).astype(int)
auc = roc_auc_score(y_te, p_te)
aucpr = average_precision_score(y_te, p_te)
f1 = f1_score(y_te, pred_te)
cm = confusion_matrix(y_te, pred_te)

with open("outputs/metrics.txt", "w") as f:
    f.write(f"Rows: {len(df)}  Default rate: {y.mean():.3%}\n")
    f.write(f"Best iteration: {model.best_iteration}\nThreshold (validation-chosen): {best_t:.2f}\n")
    f.write(f"TEST ROC-AUC: {auc:.3f}\nTEST AUCPR: {aucpr:.3f}\nTEST F1: {f1:.3f}\n")
    f.write(f"Confusion matrix [[TN FP],[FN TP]]:\n{cm}\n")
print(open("outputs/metrics.txt").read())

ConfusionMatrixDisplay(cm, display_labels=["Paid", "Default"]).plot(cmap="Blues")
plt.title("Test confusion matrix"); plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
RocCurveDisplay.from_predictions(y_te, p_te)
plt.title("Test ROC curve"); plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------- 7. Score every customer for the Power BI dashboard ----------
out = df.copy()
out["default_probability"] = model.predict_proba(X)[:, 1]
out["risk_band"] = pd.cut(out["default_probability"], [0, 0.2, 0.4, 0.6, 1.0],
                          labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
# Synthetic "portfolio_manager" column so you can demo Power BI row-level security (role-based views)
out["portfolio_manager"] = np.random.default_rng(RANDOM_STATE).choice(["Region_A", "Region_B", "Region_C"], len(out))
out.to_csv("outputs/scored_customers.csv", index_label="customer_id")
print("Saved outputs/scored_customers.csv for Power BI")

# ---------- Power BI steps (manual) ----------
# 1. Get Data -> Text/CSV -> outputs/scored_customers.csv
# 2. Visuals: default rate by risk_band, utilization vs default, months_delinquent distribution,
#    expected loss proxy = default_probability * BILL_AMT1 by risk_band.
# 3. Modeling -> Manage roles -> role "Region_A" with DAX filter: [portfolio_manager] = "Region_A"
#    (repeat for B, C). Test with "View as" to show role-based access.
