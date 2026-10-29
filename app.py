"""Step 5: Streamlit app — Real Estate Investment Advisor.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import (AMENITY_LIST, CURRENT_YEAR, HORIZON_YEARS, MODEL_CATEGORICAL,  # noqa: E402
                      MODEL_NUMERIC, create_targets, engineer_features)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# ----------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    clf = joblib.load(ROOT / "models" / "classifier.joblib")
    reg = joblib.load(ROOT / "models" / "regressor.joblib")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    lookups = json.loads((ROOT / "models" / "lookups.json").read_text())
    return clf, reg, meta, lookups


@st.cache_data
def load_data():
    path = ROOT / "data" / "processed" / "housing_clean.csv"
    if not path.exists():                       # e.g. fresh clone / Streamlit Cloud
        import preprocess
        with st.spinner("Preparing data for first run..."):
            preprocess.main()
    return pd.read_csv(path)


clf, reg, meta, lookups = load_models()
df = load_data()

st.sidebar.title("🏠 Investment Advisor")
page = st.sidebar.radio("Go to", ["Predict a property", "Explore the market", "Model insights"])
st.sidebar.caption(f"Classifier: {meta['best_classifier']}  \nRegressor: {meta['best_regressor']}")
st.sidebar.caption(f"Dataset: {len(df):,} properties · {df['City'].nunique()} cities")


# ================================================================ PREDICT
if page == "Predict a property":
    st.title("Is this property a good investment?")
    st.write("Enter the property details to get an investment verdict and a 5-year price forecast.")

    # Location pickers sit outside the form so City/Locality update when State changes
    c1, c2, c3 = st.columns(3)
    states = sorted(lookups["state_city"])
    state = c1.selectbox("State", states, index=states.index("Rajasthan") if "Rajasthan" in states else 0)
    city = c2.selectbox("City", lookups["state_city"][state])
    locality = c3.selectbox("Locality", lookups["city_localities"][city])

    with st.form("property"):
        a, b, c, d = st.columns(4)
        property_type = a.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        bhk = b.number_input("BHK", 1, 5, 3)
        size = c.number_input("Size (sq ft)", 500, 5000, 1500, step=50)
        price = d.number_input("Current price (₹ lakhs)", 10.0, 500.0, 120.0, step=5.0)

        a, b, c, d = st.columns(4)
        year_built = a.number_input("Year built", 1990, CURRENT_YEAR, 2015)
        total_floors = b.number_input("Total floors in building", 1, 30, 12)
        floor_no = c.number_input("Floor number", 0, 30, 5)
        furnished = d.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])

        a, b, c, d = st.columns(4)
        schools = a.slider("Nearby schools", 1, 10, 6)
        hospitals = b.slider("Nearby hospitals", 1, 10, 6)
        transport = c.selectbox("Public transport access", ["Low", "Medium", "High"], index=2)
        facing = d.selectbox("Facing", ["North", "East", "South", "West"])

        a, b, c, d = st.columns(4)
        parking = a.radio("Parking", ["Yes", "No"], horizontal=True)
        security = b.radio("Security (gated/CCTV)", ["Yes", "No"], horizontal=True)
        owner = c.selectbox("Listed by", ["Owner", "Builder", "Broker"])
        availability = d.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        amenities = st.multiselect("Amenities", AMENITY_LIST, default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse property", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "State": state, "City": city, "Locality": locality, "Property_Type": property_type,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year_built,
            "Furnished_Status": furnished, "Floor_No": floor_no, "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking,
            "Security": security, "Amenities": ", ".join(amenities), "Facing": facing,
            "Owner_Type": owner, "Availability_Status": availability,
        }])
        feats = engineer_features(row, lookups["city_median_ppsf"])
        X = feats[MODEL_NUMERIC + MODEL_CATEGORICAL]
        proba = float(clf.predict_proba(X)[0, 1])
        is_good = proba >= 0.5
        future = float(reg.predict(X)[0])
        cagr = (future / price) ** (1 / HORIZON_YEARS) - 1
        rules = create_targets(feats, lookups["median_growth_rate"]).iloc[0]

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Good investment?", "✅ Yes" if is_good else "❌ No")
        m2.metric("Model confidence", f"{max(proba, 1 - proba):.0%}",
                  help=f"Probability of 'Good Investment' = {proba:.1%}")
        m3.metric("Estimated price in 5 years", f"₹{future:,.1f} L", f"+₹{future - price:,.1f} L")
        m4.metric("Implied annual growth", f"{cagr:.1%}")

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Why this verdict")
            city_med = lookups["city_median_ppsf"][city]
            checks = [
                (rules["Rule_Value_Buy"], "Value buy",
                 f"₹{feats['Price_per_SqFt_Rs'].iloc[0]:,.0f}/sq ft vs {city} median ₹{city_med:,.0f}"),
                (rules["Rule_High_Growth"], "High growth potential",
                 f"rule-based growth {rules['Annual_Growth_Rate']:.1%}/yr vs median "
                 f"{lookups['median_growth_rate']:.1%}"),
                (rules["Rule_Good_Infra"], "Good social infrastructure",
                 f"{schools + hospitals} schools + hospitals nearby (threshold 11)"),
                (rules["Rule_Connected"], "Well connected", f"public transport: {transport}"),
            ]
            for ok, title, detail in checks:
                st.markdown(f"{'✅' if ok else '❌'} **{title}** — {detail}")
            st.caption(f"Meets {int(rules['Investment_Score'])} of 4 rules; 3 or more = Good Investment. "
                       "The verdict above is the model's prediction.")
        with right:
            years = np.arange(0, HORIZON_YEARS + 1)
            path = pd.DataFrame({"Year": CURRENT_YEAR + years, "Price (₹ lakhs)": price * (1 + cagr) ** years})
            fig = px.line(path, x="Year", y="Price (₹ lakhs)", markers=True, title="Projected price path")
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("What the models pay attention to")
        i1, i2 = st.columns(2)
        for col, key, title in [(i1, "clf_importance", "Investment classifier"),
                                (i2, "reg_importance", "Price regressor")]:
            imp = pd.Series(meta[key]).head(10)[::-1]
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", title=title,
                         labels={"x": "Relative importance", "y": ""})
            fig.update_layout(height=360, margin=dict(t=40, b=10))
            col.plotly_chart(fig, use_container_width=True)


# ================================================================ EXPLORE
elif page == "Explore the market":
    st.title("Explore the market")
    with st.sidebar:
        st.subheader("Filters")
        f_states = st.multiselect("State", sorted(df["State"].unique()))
        city_opts = sorted(df[df["State"].isin(f_states)]["City"].unique()) if f_states else sorted(df["City"].unique())
        f_cities = st.multiselect("City", city_opts)
        f_types = st.multiselect("Property type", sorted(df["Property_Type"].unique()))
        f_bhk = st.slider("BHK", 1, 5, (1, 5))
        f_price = st.slider("Price (₹ lakhs)", 10, 500, (10, 500))
        f_size = st.slider("Size (sq ft)", 500, 5000, (500, 5000), step=100)
        f_good = st.checkbox("Only Good Investments")

    d = df
    if f_states: d = d[d["State"].isin(f_states)]
    if f_cities: d = d[d["City"].isin(f_cities)]
    if f_types: d = d[d["Property_Type"].isin(f_types)]
    d = d[d["BHK"].between(*f_bhk) & d["Price_in_Lakhs"].between(*f_price) & d["Size_in_SqFt"].between(*f_size)]
    if f_good: d = d[d["Good_Investment"] == 1]

    if d.empty:
        st.warning("No properties match these filters.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Properties", f"{len(d):,}")
    k2.metric("Median price", f"₹{d['Price_in_Lakhs'].median():,.0f} L")
    k3.metric("Median ₹/sq ft", f"₹{d['Price_per_SqFt_Rs'].median():,.0f}")
    k4.metric("Good investments", f"{d['Good_Investment'].mean():.0%}")

    metric = st.radio("Heatmap shows", ["% Good Investment", "Median ₹/sq ft", "Avg 5-yr appreciation"],
                      horizontal=True)
    val = {"% Good Investment": ("Good_Investment", "mean", 100),
           "Median ₹/sq ft": ("Price_per_SqFt_Rs", "median", 1),
           "Avg 5-yr appreciation": ("_appr", "mean", 100)}[metric]
    d = d.assign(_appr=d["Future_Price_5Y"] / d["Price_in_Lakhs"] - 1)
    hm = d.pivot_table(index="City", columns="Property_Type", values=val[0], aggfunc=val[1]) * val[2]
    fig = px.imshow(hm.round(1), text_auto=True, aspect="auto", color_continuous_scale="Teal",
                    title=f"{metric} by city and property type")
    fig.update_layout(height=max(350, 22 * len(hm)), margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    by_bhk = d.groupby("BHK").agg(price=("Price_in_Lakhs", "median"), good=("Good_Investment", "mean")).reset_index()
    fig = px.bar(by_bhk, x="BHK", y="price", title="Median price by BHK", labels={"price": "₹ lakhs"})
    c1.plotly_chart(fig, use_container_width=True)
    by_age = d.groupby("Year_Built").agg(appr=("_appr", "mean")).reset_index()
    fig = px.line(by_age, x="Year_Built", y="appr", markers=True, title="Expected 5-yr appreciation by year built",
                  labels={"appr": "Appreciation", "Year_Built": "Year built"})
    fig.update_yaxes(tickformat=".0%")
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Top-ranked properties")
    cols = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_Rs", "Future_Price_5Y", "Investment_Score", "Good_Investment"]
    top = d.sort_values(["Investment_Score", "Annual_Growth_Rate"], ascending=False)[cols].head(200)
    st.dataframe(top.round(1), use_container_width=True, hide_index=True)
    st.download_button("Download filtered data (CSV)", d.drop(columns="_appr").to_csv(index=False),
                       "filtered_properties.csv", "text/csv")


# ================================================================ INSIGHTS
else:
    st.title("Model insights")
    st.write("Every model was trained on 80% of the data and tested on the remaining 50,000 properties. "
             "All runs are tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).")
    c = pd.DataFrame(meta["results"]["classification"]).set_index("model")
    r = pd.DataFrame(meta["results"]["regression"]).set_index("model")
    st.subheader("Classification — Good Investment")
    st.dataframe(c.style.format({"train_seconds": "{:.0f}", **{k: "{:.4f}" for k in c.columns if k != "train_seconds"}})
                 .highlight_max(subset=["accuracy", "f1", "roc_auc"], props="background-color: #1f7a4d; color: #ffffff; font-weight: 600;"), use_container_width=True)
    st.subheader("Regression — price after 5 years (₹ lakhs)")
    st.dataframe(r.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.5f}", "mape": "{:.2%}",
                                 "train_seconds": "{:.0f}"})
                 .highlight_min(subset=["rmse", "mae"], props="background-color: #1f7a4d; color: #ffffff; font-weight: 600;"), use_container_width=True)

    figdir = ROOT / "reports" / "figures" / "models"
    best_c, best_r = meta["best_classifier"], meta["best_regressor"]
    a, b, c3 = st.columns(3)
    a.image(str(figdir / f"cm_{best_c}.png"), caption=f"Confusion matrix — {best_c}")
    b.image(str(figdir / f"importance_{best_c}.png"), caption="Classifier feature importance")
    c3.image(str(figdir / f"actual_vs_pred_{best_r}.png"), caption=f"Actual vs predicted — {best_r}")

    st.info("**Read these scores in context.** The dataset has no real investment outcomes, so both targets "
            "were built from documented domain rules (see README). The high scores show the models learned "
            "those rules accurately — not that they can predict real-world market returns.")
