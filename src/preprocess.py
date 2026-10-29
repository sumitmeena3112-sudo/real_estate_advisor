"""Step 1: data cleaning, outlier detection, feature engineering, target creation.

Run:  python src/preprocess.py
Output: data/processed/housing_clean.csv, models/lookups.json, reports/data_quality.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import create_targets, engineer_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india_housing_prices.csv"
OUT = ROOT / "data" / "processed" / "housing_clean.csv"
LOOKUPS = ROOT / "models" / "lookups.json"
REPORT = ROOT / "reports" / "data_quality.md"


def main():
    df = pd.read_csv(RAW)
    log = [f"# Data quality report\n", f"- Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns"]

    # 1. Duplicates ---------------------------------------------------------
    n_dup_full = df.duplicated().sum()
    n_dup_noid = df.drop(columns="ID").duplicated().sum()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    log.append(f"- Exact duplicate rows: {n_dup_full}; duplicates ignoring ID: {n_dup_noid} (removed)")

    # 2. Missing values -----------------------------------------------------
    missing = df.isna().sum()
    log.append(f"- Missing values: {int(missing.sum())} in total")
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())          # median imputation
    for c in cat_cols:                                                   # mode imputation
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    df["Amenities"] = df["Amenities"].fillna("")

    # 3. Feature engineering ------------------------------------------------
    df["_ppsf"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    city_median_ppsf = df.groupby("City")["_ppsf"].median().round(2).to_dict()
    df = df.drop(columns="_ppsf")
    df = engineer_features(df, city_median_ppsf)
    n_floor = int(df["Floor_Inconsistent"].sum())
    log.append(f"- Floor_No greater than Total_Floors: {n_floor:,} rows "
               f"({n_floor / len(df):.1%}); flagged and capped at Total_Floors")

    # 4. Outlier detection (IQR rule) ---------------------------------------
    log.append("\n## Outliers (IQR rule: outside Q1 - 1.5*IQR, Q3 + 1.5*IQR)\n")
    log.append("| Column | Lower fence | Upper fence | Outliers | Share |")
    log.append("|---|---|---|---|---|")
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_Rs"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = (df[col] < lo) | (df[col] > hi)
        df[f"Outlier_{col}"] = mask.astype(int)
        log.append(f"| {col} | {lo:,.2f} | {hi:,.2f} | {mask.sum():,} | {mask.mean():.1%} |")
    log.append("\nOutliers are flagged, not removed: Price_per_SqFt is a ratio of two "
               "independent, uniformly spread columns, so its long right tail comes from "
               "small homes with high prices — genuine records rather than entry errors. "
               "Tree models are robust to them, and linear models use scaled features.")

    # 5. Targets ------------------------------------------------------------
    from features import appreciation_rate
    median_rate = float(appreciation_rate(df).median())
    df = create_targets(df, median_rate)
    log.append("\n## Targets\n")
    log.append(f"- Good_Investment share: {df['Good_Investment'].mean():.1%} positive")
    log.append(f"- Annual growth rate: min {df['Annual_Growth_Rate'].min():.2%}, "
               f"median {median_rate:.2%}, max {df['Annual_Growth_Rate'].max():.2%}")
    log.append(f"- Future_Price_5Y (lakhs): mean {df['Future_Price_5Y'].mean():.1f}, "
               f"range {df['Future_Price_5Y'].min():.1f}-{df['Future_Price_5Y'].max():.1f}")
    log.append(f"\n- Clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    LOOKUPS.parent.mkdir(parents=True, exist_ok=True)
    LOOKUPS.write_text(json.dumps({
        "city_median_ppsf": city_median_ppsf,
        "median_growth_rate": median_rate,
        "state_city": df.groupby("State")["City"].unique().apply(sorted).apply(list).to_dict(),
        "city_localities": df.groupby("City")["Locality"].unique().apply(sorted).apply(list).to_dict(),
    }, indent=1))
    REPORT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
