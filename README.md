# Credit Card Default Risk Model

Predicts which credit card customers will default next month, then scores every customer into risk bands for reporting (Power BI-ready output).

**Data:** UCI "Default of Credit Card Clients" — 30,000 cardholders, 22.1% default rate ([Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)). Save as `UCI_Credit_Card.csv`.

## Approach
- Behavioral features: credit utilization (latest and 6-month average), payment ratio, months delinquent, max delinquency, bill trend
- Stratified 70 / 15 / 15 train / validation / test split
- XGBoost with early stopping on validation (AUCPR)
- Decision threshold chosen on **validation only** (max F1); test set evaluated once

## Results (locked test set, 4,500 customers)
| Metric | Value |
|---|---|
| ROC-AUC | **0.783** |
| AUCPR | 0.558 |
| F1 | 0.538 |
| Threshold (validation-chosen) | 0.28 |
| Confusion matrix [[TN FP],[FN TP]] | [[2985 519],[438 558]] |

## Calibration
Ranking customers well (ROC-AUC) is not the same as predicting the right default *rate*. The script also checks whether predicted probabilities match observed default rates on the test set:
- **Brier score** vs. a base-rate model (always predict the training default rate)
- **Expected calibration error** over 10 quantile bins
- Mean predicted vs. observed default rate, and a calibration curve (`outputs/calibration_curve.png`)

## Run
```bash
pip install pandas scikit-learn xgboost matplotlib
python credit_card_default_project.py
```
Outputs `outputs/metrics.txt` (including calibration), ROC, confusion-matrix and calibration plots, and `scored_customers.csv` (risk band per customer).
