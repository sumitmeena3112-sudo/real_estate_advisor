# Model results

Test set: 50,000 properties (20% stratified hold-out).

## Classification — Good_Investment

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.8867 | 0.8655 | 0.8627 | 0.8641 | 0.9608 | 1 |
| RandomForestClassifier | 0.9620 | 0.9546 | 0.9542 | 0.9544 | 0.9950 | 34 |
| XGBClassifier | 0.9942 | 0.9930 | 0.9931 | 0.9931 | 0.9998 | 11 |

Selected: **XGBClassifier** (highest F1).

## Regression — Future_Price_5Y (lakhs)

| Model | RMSE | MAE | R² | MAPE | Train time (s) |
|---|---|---|---|---|---|
| LinearRegression | 9.516 | 6.653 | 0.99816 | 4.64% | 2 |
| RandomForestRegressor | 6.470 | 4.654 | 0.99915 | 1.27% | 435 |
| XGBRegressor | 1.736 | 1.351 | 0.99994 | 0.54% | 11 |

Selected: **XGBRegressor** (lowest RMSE).
