"""Shared feature engineering and target-creation logic.

Used by preprocess.py (whole dataset) and app.py (single property from the form),
so the app always transforms inputs exactly the way the models were trained.
"""
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025          # Age_of_Property in the dataset = 2025 - Year_Built
HORIZON_YEARS = 5
AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

# ---------------------------------------------------------------------------
# Domain rules for the 5-year appreciation rate (documented in README)
# ---------------------------------------------------------------------------
BASE_ANNUAL_GROWTH = 0.08    # long-run residential appreciation assumption
TRANSPORT_ADJ = {"High": 0.010, "Medium": 0.005, "Low": 0.0}
PROPERTY_TYPE_ADJ = {"Villa": 0.005, "Independent House": 0.003, "Apartment": 0.0}
FURNISH_ADJ = {"Furnished": 0.002, "Semi-furnished": 0.001, "Unfurnished": 0.0}

MODEL_CATEGORICAL = ["State", "City", "Property_Type", "Furnished_Status",
                     "Public_Transport_Accessibility", "Facing", "Owner_Type",
                     "Availability_Status"]
MODEL_NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs",
                 "Age_of_Property", "Floor_No", "Total_Floors", "Floor_Ratio",
                 "Nearby_Schools", "Nearby_Hospitals", "Infra_Score",
                 "School_Density_Score", "Transport_Score", "Amenity_Count",
                 "Parking", "Has_Security", "PPSF_vs_City_Median"] + \
                [f"Has_{a}" for a in AMENITY_LIST]


def engineer_features(df: pd.DataFrame, city_median_ppsf: dict) -> pd.DataFrame:
    """Add engineered features. Works on the full dataset or a 1-row frame."""
    out = df.copy()

    # Precise price per sq ft in rupees (raw column is lakhs/sqft rounded to 2 dp)
    out["Price_per_SqFt_Rs"] = out["Price_in_Lakhs"] * 1e5 / out["Size_in_SqFt"]

    out["Age_of_Property"] = CURRENT_YEAR - out["Year_Built"]

    # Floor sanity: Floor_No above Total_Floors is impossible -> flag and cap
    out["Floor_Inconsistent"] = (out["Floor_No"] > out["Total_Floors"]).astype(int)
    out["Floor_No"] = np.minimum(out["Floor_No"], out["Total_Floors"])
    out["Floor_Ratio"] = out["Floor_No"] / out["Total_Floors"].clip(lower=1)

    # Infrastructure features
    out["Infra_Score"] = out["Nearby_Schools"] + out["Nearby_Hospitals"]
    out["School_Density_Score"] = out["Nearby_Schools"] / 10          # 0.1-1.0
    out["Transport_Score"] = out["Public_Transport_Accessibility"].map(
        {"Low": 1, "Medium": 2, "High": 3})

    # Amenities: multi-valued text -> one flag per amenity + count
    amen = out["Amenities"].fillna("")
    for a in AMENITY_LIST:
        out[f"Has_{a}"] = amen.str.contains(a, regex=False).astype(int)
    out["Amenity_Count"] = out[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)

    out["Parking"] = (out["Parking_Space"] == "Yes").astype(int)
    out["Has_Security"] = (out["Security"] == "Yes").astype(int)

    # Relative value: how this property's rate compares with its city
    city_med = out["City"].map(city_median_ppsf)
    city_med = city_med.fillna(np.median(list(city_median_ppsf.values())))
    out["PPSF_vs_City_Median"] = out["Price_per_SqFt_Rs"] / city_med
    return out


def appreciation_rate(df: pd.DataFrame) -> pd.Series:
    """Rule-based expected annual appreciation for each property."""
    r = pd.Series(BASE_ANNUAL_GROWTH, index=df.index)
    r += df["Public_Transport_Accessibility"].map(TRANSPORT_ADJ).fillna(0)
    r += df["Property_Type"].map(PROPERTY_TYPE_ADJ).fillna(0)
    r += df["Furnished_Status"].map(FURNISH_ADJ).fillna(0)
    r += (df["Infra_Score"] - 11) / 18 * 0.010           # about -0.5% to +0.5%
    r += np.where(df["Availability_Status"] == "Under_Construction", 0.010, 0.0)
    r -= 0.0005 * df["Age_of_Property"]                    # older = slower growth
    r += 0.002 * df["Amenity_Count"]
    r += 0.003 * df["Has_Security"] + 0.002 * df["Parking"]
    return r


def create_targets(df: pd.DataFrame, median_rate: float) -> pd.DataFrame:
    """Create both target variables (they do not exist in the raw data)."""
    out = df.copy()
    out["Annual_Growth_Rate"] = appreciation_rate(out)
    out["Future_Price_5Y"] = out["Price_in_Lakhs"] * (1 + out["Annual_Growth_Rate"]) ** HORIZON_YEARS

    # Good Investment = meets at least 3 of these 4 rules
    out["Rule_Value_Buy"] = (out["PPSF_vs_City_Median"] <= 1.0).astype(int)
    out["Rule_High_Growth"] = (out["Annual_Growth_Rate"] >= median_rate).astype(int)
    out["Rule_Good_Infra"] = (out["Infra_Score"] >= 11).astype(int)
    out["Rule_Connected"] = (out["Transport_Score"] >= 2).astype(int)
    rules = ["Rule_Value_Buy", "Rule_High_Growth", "Rule_Good_Infra", "Rule_Connected"]
    out["Investment_Score"] = out[rules].sum(axis=1)
    out["Good_Investment"] = (out["Investment_Score"] >= 3).astype(int)
    return out
