"""Step 2: Exploratory Data Analysis — the 20 questions from the brief + approach-level EDA.

Run:  python src/eda.py
Output: reports/figures/*.png and reports/eda_findings.md (numbers computed from the data)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "housing_clean.csv"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "eda_findings.md"

sns.set_theme(style="whitegrid", palette="crest")
FIG.mkdir(parents=True, exist_ok=True)
findings = ["# EDA findings\n",
            "All numbers below are computed by `src/eda.py` from the cleaned data. "
            "Price per sq ft is in ₹ (`Price_per_SqFt_Rs`), prices in lakhs.\n"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    return f"![{name}](figures/{name})"


def section(title):
    findings.append(f"\n## {title}\n")


def q(num, question, answer, img=None):
    findings.append(f"**Q{num}. {question}**  \n{answer}\n")
    if img:
        findings.append(img + "\n")


def spread(series):
    """Describe how much a grouped mean varies: max/min gap in %."""
    return (series.max() - series.min()) / series.min() * 100


df = pd.read_csv(DATA)
sample = df.sample(20000, random_state=42)

# ---------------------------------------------------------------- 1-5
section("1-5: Price & size analysis")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax)
ax.set(title="Distribution of property prices", xlabel="Price (lakhs)")
p = df["Price_in_Lakhs"]
q(1, "What is the distribution of property prices?",
  f"Prices range from ₹{p.min():.0f}L to ₹{p.max():.0f}L with mean ₹{p.mean():.0f}L and median "
  f"₹{p.median():.0f}L. Skewness is {p.skew():.2f}: the distribution is almost flat (uniform) "
  f"rather than the right-skewed shape typical of real housing markets.",
  save(fig, "q01_price_distribution.png"))

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax)
ax.set(title="Distribution of property sizes", xlabel="Size (sq ft)")
s = df["Size_in_SqFt"]
q(2, "What is the distribution of property sizes?",
  f"Sizes range from {s.min():,} to {s.max():,} sq ft (median {s.median():,.0f}). "
  f"Like price, size is spread almost evenly across its range (skew {s.skew():.2f}).",
  save(fig, "q02_size_distribution.png"))

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_Rs", showfliers=False, ax=ax)
ax.set(title="Price per sq ft by property type", ylabel="₹ per sq ft")
g = df.groupby("Property_Type")["Price_per_SqFt_Rs"].median()
q(3, "How does price per sq ft vary by property type?",
  "Median ₹/sq ft: " + ", ".join(f"{k} ₹{v:,.0f}" for k, v in g.items()) +
  f". The gap between types is only {spread(g):.1f}%, so property type barely changes the rate.",
  save(fig, "q03_ppsf_by_type.png"))

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", s=6, alpha=0.3, ax=ax)
ax.set(title="Size vs price (20k sample)")
r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
q(4, "Is there a relationship between property size and price?",
  f"No. The correlation is {r:.3f}: bigger homes are not more expensive in this dataset. "
  "This is the most important finding of the EDA — in a real market this correlation is "
  "strongly positive. It suggests the dataset is synthetic, with columns generated independently.",
  save(fig, "q04_size_vs_price.png"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["Price_per_SqFt_Rs"], ax=axes[0]); axes[0].set_title("Price per sq ft (₹)")
sns.boxplot(y=df["Size_in_SqFt"], ax=axes[1]); axes[1].set_title("Size (sq ft)")
n_out = df["Outlier_Price_per_SqFt_Rs"].sum()
q(5, "Are there any outliers in price per sq ft or property size?",
  f"Price per sq ft: {n_out:,} outliers ({n_out / len(df):.1%}) above the IQR upper fence, "
  "all on the high side (small homes with high prices). Size: none — it is bounded between "
  "500 and 5,000 sq ft. Outliers were flagged, not dropped (see data_quality.md).",
  save(fig, "q05_outliers.png"))

# ---------------------------------------------------------------- 6-10
section("6-10: Location-based analysis")

g = df.groupby("State")["Price_per_SqFt_Rs"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=g.values, y=g.index, ax=ax, color="#3a7ca5")
ax.set(title="Average price per sq ft by state", xlabel="₹ per sq ft")
q(6, "What is the average price per sq ft by state?",
  f"Highest: {g.index[0]} (₹{g.iloc[0]:,.0f}); lowest: {g.index[-1]} (₹{g.iloc[-1]:,.0f}). "
  f"The spread across all 20 states is only {spread(g):.1f}%.",
  save(fig, "q06_ppsf_by_state.png"))

g = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 10))
sns.barplot(x=g.values, y=g.index, ax=ax, color="#3a7ca5")
ax.set(title="Average property price by city", xlabel="Price (lakhs)")
q(7, "What is the average property price by city?",
  f"Highest: {g.index[0]} (₹{g.iloc[0]:.1f}L); lowest: {g.index[-1]} (₹{g.iloc[-1]:.1f}L). "
  f"Across 42 cities the averages differ by just {spread(g):.1f}% — location does not drive "
  "price here.",
  save(fig, "q07_price_by_city.png"))

g = df.groupby("Locality")["Age_of_Property"].median()
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(g, bins=20, ax=ax)
ax.set(title="Median property age across 500 localities", xlabel="Median age (years)")
q(8, "What is the median age of properties by locality?",
  f"Locality medians range from {g.min():.0f} to {g.max():.0f} years (overall median "
  f"{df['Age_of_Property'].median():.0f}). The histogram shows the distribution across all "
  "500 localities; the full table is easy to produce with `groupby('Locality')`.",
  save(fig, "q08_age_by_locality.png"))

ct = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(ct, annot=True, fmt=".0f", cmap="crest", ax=ax, cbar_kws={"label": "% of city listings"})
ax.set(title="BHK mix by city (%)")
q(9, "How is BHK distributed across cities?",
  f"Every city has roughly 20% of listings in each BHK from 1 to 5 (range "
  f"{ct.values.min():.1f}%-{ct.values.max():.1f}%). No city skews toward small or large homes.",
  save(fig, "q09_bhk_by_city.png"))

top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
t = df[df["Locality"].isin(top5)].copy()
t["Built_Period"] = (t["Year_Built"] // 5) * 5
trend = t.groupby(["Built_Period", "Locality"])["Price_in_Lakhs"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(data=trend, x="Built_Period", y="Price_in_Lakhs", hue="Locality", marker="o", ax=ax)
ax.set(title="Top 5 most expensive localities: price by construction period",
       xlabel="Year built (5-year bins)", ylabel="Avg price (lakhs)")
q(10, "What are the price trends for the top 5 most expensive localities?",
  f"Top 5 by average price: {', '.join(top5)}. The data has no sale dates, so the trend uses "
  "construction year (5-year bins) as the time axis. Prices fluctuate between periods with no "
  "consistent upward or downward direction.",
  save(fig, "q10_top5_locality_trend.png"))

# ---------------------------------------------------------------- 11-15
section("11-15: Feature relationships & correlation")

num = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_Rs", "BHK", "Age_of_Property",
       "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count",
       "Transport_Score", "Future_Price_5Y", "Good_Investment"]
corr = df[num].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set(title="Correlation of numeric features")
raw_cols = ["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Age_of_Property", "Floor_No",
            "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Transport_Score"]
rc = corr.loc[raw_cols, raw_cols].where(~np.eye(len(raw_cols), dtype=bool)).abs()
pair = rc.stack().idxmax()
q(11, "How are numeric features correlated with each other?",
  f"The raw features are essentially uncorrelated: the strongest pair is {pair[0]} vs "
  f"{pair[1]} at |r| = {rc.loc[pair]:.3f}. Floor_No vs Total_Floors is the exception "
  f"(r = {corr.loc['Floor_No', 'Total_Floors']:.2f}), caused by capping Floor_No during cleaning. "
  "The only strong correlations are derived ones: price per sq ft vs "
  f"size (r = {corr.loc['Size_in_SqFt', 'Price_per_SqFt_Rs']:.2f}) and price "
  f"(r = {corr.loc['Price_in_Lakhs', 'Price_per_SqFt_Rs']:.2f}), and future price vs "
  f"current price (r = {corr.loc['Price_in_Lakhs', 'Future_Price_5Y']:.2f}).",
  save(fig, "q11_correlation_heatmap.png"))

for n, col, label in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
    g = df.groupby(col)["Price_per_SqFt_Rs"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=g.index, y=g.values, marker="o", ax=ax)
    ax.set(title=f"Avg price per sq ft by number of nearby {label}", xlabel=col, ylabel="₹ per sq ft")
    ax.set_ylim(0, g.max() * 1.2)
    q(n, f"How do nearby {label} relate to price per sq ft?",
      f"Correlation {df[col].corr(df['Price_per_SqFt_Rs']):.3f}. The average rate stays "
      f"between ₹{g.min():,.0f} and ₹{g.max():,.0f} for 1 to 10 {label} — no premium for more "
      f"{label} in current prices. (In the investment rules they do raise expected growth.)",
      save(fig, f"q{n}_{label}_vs_ppsf.png"))

g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax)
ax.set(title="Price by furnished status")
q(14, "How does price vary by furnished status?",
  "Mean price: " + ", ".join(f"{k} ₹{v:.1f}L" for k, v in g.items()) +
  f" — a {spread(g):.1f}% spread, so furnishing has no visible effect on price.",
  save(fig, "q14_price_by_furnishing.png"))

g = df.groupby("Facing")["Price_per_SqFt_Rs"].mean()
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=g.index, y=g.values, ax=ax, color="#3a7ca5")
ax.set(title="Avg price per sq ft by facing direction", ylabel="₹ per sq ft")
q(15, "How does price per sq ft vary by property facing direction?",
  "Mean ₹/sq ft: " + ", ".join(f"{k} ₹{v:,.0f}" for k, v in g.items()) +
  f" ({spread(g):.1f}% spread). Facing direction does not command a premium.",
  save(fig, "q15_ppsf_by_facing.png"))

# ---------------------------------------------------------------- 16-20
section("16-20: Investment, amenities & ownership")

g = df["Owner_Type"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=g.index, y=g.values, ax=ax, color="#3a7ca5")
ax.set(title="Properties by owner type", ylabel="Count")
q(16, "How many properties belong to each owner type?",
  ", ".join(f"{k}: {v:,}" for k, v in g.items()) + " — an even three-way split.",
  save(fig, "q16_owner_type.png"))

g = df["Availability_Status"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=g.index, y=g.values, ax=ax, color="#3a7ca5")
ax.set(title="Properties by availability status", ylabel="Count")
q(17, "How many properties are available under each availability status?",
  ", ".join(f"{k}: {v:,}" for k, v in g.items()) +
  ". The brief mentions a 'Sold' status, but it does not appear in the data.",
  save(fig, "q17_availability.png"))

g = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax)
ax.set(title="Price with vs without parking")
q(18, "Does parking space affect property price?",
  f"Mean price with parking ₹{g.get('Yes', np.nan):.1f}L vs without ₹{g.get('No', np.nan):.1f}L "
  f"({spread(g):.1f}% difference). Parking does not affect price in this data. "
  "(Parking_Space is Yes/No, not a count as the brief describes.)",
  save(fig, "q18_parking_vs_price.png"))

g = df.groupby("Amenity_Count")["Price_per_SqFt_Rs"].mean()
gi = {a: df.loc[df[f"Has_{a}"] == 1, "Price_per_SqFt_Rs"].mean()
      for a in ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=g.index, y=g.values, ax=axes[0], color="#3a7ca5")
axes[0].set(title="Avg ₹/sq ft by number of amenities", xlabel="Amenity count", ylabel="₹ per sq ft")
sns.barplot(x=list(gi), y=list(gi.values()), ax=axes[1], color="#81b29a")
axes[1].set(title="Avg ₹/sq ft for listings with each amenity", ylabel="₹ per sq ft")
q(19, "How do amenities affect price per sq ft?",
  f"Each listing has 1-5 amenities. Average ₹/sq ft ranges from ₹{g.min():,.0f} to "
  f"₹{g.max():,.0f} across amenity counts ({spread(g):.1f}% spread), and no single amenity "
  "carries a premium. Amenities matter for the investment score, not the current price.",
  save(fig, "q19_amenities_vs_ppsf.png"))

order = ["Low", "Medium", "High"]
g1 = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt_Rs"].mean().reindex(order)
g2 = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(order) * 100
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=g1.index, y=g1.values, ax=axes[0], color="#3a7ca5")
axes[0].set(title="Avg ₹/sq ft by transport access", ylabel="₹ per sq ft")
sns.barplot(x=g2.index, y=g2.values, ax=axes[1], color="#81b29a")
axes[1].set(title="% labelled Good Investment", ylabel="%")
q(20, "How does public transport accessibility relate to price per sq ft or investment potential?",
  "Current rate barely moves (" + ", ".join(f"{k} ₹{v:,.0f}" for k, v in g1.items()) +
  "), but investment potential does: " +
  ", ".join(f"{k} {v:.0f}%" for k, v in g2.items()) +
  " of properties are Good Investments. This is by design — transport access is one of the "
  "four investment rules and adds to expected appreciation.",
  save(fig, "q20_transport.png"))

# ---------------------------------------------------- Approach-level EDA
section("Additional analysis from the Approach section")

g = df.groupby("City")["Price_per_SqFt_Rs"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(x=g.index, y=g.values, ax=ax, color="#3a7ca5")
ax.set(title="Median price per sq ft by city", ylabel="₹ per sq ft"); ax.tick_params(axis="x", rotation=75)
findings.append("**Price trends by city.** Median ₹/sq ft ranges from "
                f"₹{g.min():,.0f} ({g.index[-1]}) to ₹{g.max():,.0f} ({g.index[0]}).\n\n"
                + save(fig, "a1_city_ppsf.png") + "\n")

df["Size_Band"] = pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000],
                         labels=["<1k", "1-2k", "2-3k", "3-4k", "4-5k"])
g = df.groupby("Size_Band", observed=True)[["Annual_Growth_Rate", "Good_Investment"]].mean()
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=g.index.astype(str), y=g["Good_Investment"] * 100, ax=ax, color="#81b29a")
ax.set(title="% Good Investment by size band", xlabel="Size (sq ft)", ylabel="%")
findings.append("**Area vs investment return.** Share of Good Investments by size: " +
                ", ".join(f"{i} {v:.0%}" for i, v in g["Good_Investment"].items()) +
                ". Larger homes have a lower ₹/sq ft and so pass the 'value buy' rule more often.\n\n"
                + save(fig, "a2_size_vs_investment.png") + "\n")

findings.append("**Crime rate vs Good Investment.** The brief asks for this, but the dataset has "
                "no crime-rate column, so it cannot be analysed. The closest proxy is `Security` "
                f"(gated/CCTV): {df.loc[df.Has_Security == 1, 'Good_Investment'].mean():.0%} of "
                "properties with security are Good Investments vs "
                f"{df.loc[df.Has_Security == 0, 'Good_Investment'].mean():.0%} without.\n")

g = df.groupby("Infra_Score")["Future_Price_5Y"].mean() / df.groupby("Infra_Score")["Price_in_Lakhs"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=g.index, y=(g.values - 1) * 100, marker="o", ax=ax)
ax.set(title="Expected 5-year appreciation by infrastructure score",
       xlabel="Infra score (schools + hospitals)", ylabel="5-yr appreciation (%)")
findings.append("**Infrastructure vs resale value.** Expected 5-year appreciation rises from "
                f"{(g.iloc[0] - 1):.0%} at the lowest infrastructure score to "
                f"{(g.iloc[-1] - 1):.0%} at the highest.\n\n" + save(fig, "a3_infra_vs_resale.png") + "\n")

findings.append("\n## Key takeaway\n\nThe raw features are statistically independent of price: "
                "size, location, amenities and BHK show no relationship with current price. "
                "The dataset appears to be synthetic. Investment value in this project therefore comes from "
                "the domain rules used to build the targets (see README), and the models learn "
                "those rules — this should be stated openly when presenting the project.\n")

REPORT.write_text("\n".join(findings))
print(f"Wrote {REPORT} and {len(list(FIG.glob('*.png')))} figures")
