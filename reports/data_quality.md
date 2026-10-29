# Data quality report

- Raw shape: 250,000 rows x 23 columns
- Exact duplicate rows: 0; duplicates ignoring ID: 0 (removed)
- Missing values: 0 in total
- Floor_No greater than Total_Floors: 116,304 rows (46.5%); flagged and capped at Total_Floors

## Outliers (IQR rule: outside Q1 - 1.5*IQR, Q3 + 1.5*IQR)

| Column | Lower fence | Upper fence | Outliers | Share |
|---|---|---|---|---|
| Price_in_Lakhs | -233.94 | 743.38 | 0 | 0.0% |
| Size_in_SqFt | -1,753.50 | 7,250.50 | 0 | 0.0% |
| Price_per_SqFt_Rs | -11,973.98 | 32,764.21 | 19,723 | 7.9% |

Outliers are flagged, not removed: Price_per_SqFt is a ratio of two independent, uniformly spread columns, so its long right tail comes from small homes with high prices — genuine records rather than entry errors. Tree models are robust to them, and linear models use scaled features.

## Targets

- Good_Investment share: 41.7% positive
- Annual growth rate: min 6.11%, median 9.29%, max 12.54%
- Future_Price_5Y (lakhs): mean 397.3, range 14.1-884.2

- Clean shape: 250,000 rows x 49 columns
