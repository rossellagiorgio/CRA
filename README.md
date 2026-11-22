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

## Results 

For these classification models, we focus on recall, F1-score, and AUC-ROC (rather than accuracy alone) for the default class, since the dataset is imbalanced and accuracy can be misleading. 
Logistic Regression and Gradient Boosting perform nearly identically across every metric (f1-score around 0.40, recall 64-67%, AUC around 0.70), trading lower overall accuracy (64%) for much better detection of defaulters. KNN reaches higher accuracy (77%) but misses around 83% of true defaulters (recall 0.17, f1-score 0.21) and has an AUC of only 0.58, that is, barely above random guessing. The output is reasonable, since KNN, unlike the other two models, was not trained with class balancing.

