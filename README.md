# CRA - Credit Risk Analysis

Predicting whether a loan will default, using historical data from [Lending Club in Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
The goal is to estimate the likelihood of default using only pre-origination borrower and loan data.

## Approach

- Defined a binary target from `loan_status`, keeping only loans with a definitive outcome (`Fully Paid` vs `Charged Off`/`Default`) and excluding loans still in progress (`Current`, `Late`, etc.).
- Screened categorical and numeric features for predictive signal, checking for missing values, leakage, and multicollinearity along the way.
- Trained and compared Logistic Regression, Gradient Boosting, and KNN, addressing class imbalance and evaluating on recall/F1 rather than accuracy alone.

## Project structure

```
.
├── loan.csv      # not included — see Kaggle link above
├── Credit Risk Analysis.ipynb
├── README.md
```

## How to run

Download `loan.csv` from Kaggle, place it in the project root, then open the notebook.
