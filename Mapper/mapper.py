import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import json

# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/HVI_LACounty.csv")
df.columns = df.columns.str.strip()
df["census_tract"] = df["census_tract"].astype(str)
df["hvi"] = pd.to_numeric(df["hvi"], errors="coerce")
df = df.dropna(subset=["hvi"])

with open("data/la_county.geojson") as f:
    geo = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geo["features"])
gdf.set_crs(epsg=4326, inplace=True)

# ── 2. Summary statistics ─────────────────────────────────────────────────────

print("=" * 50)
print("HVI SUMMARY STATISTICS")
print("=" * 50)
print(df["hvi"].describe().round(2))

print("\nTop 10 highest-risk census tracts:")
print(df.nlargest(10, "hvi")[["census_tract", "census_city", "hvi"]].to_string(index=False))

print("\nTop 10 lowest-risk census tracts:")
print(df.nsmallest(10, "hvi")[["census_tract", "census_city", "hvi"]].to_string(index=False))

# ── 3. City-level aggregation ─────────────────────────────────────────────────

city_stats = (
    df.groupby("census_city")["hvi"]
    .agg(["mean", "median", "max", "min", "count"])
    .rename(columns={"mean": "avg_hvi", "median": "med_hvi",
                     "max": "max_hvi", "min": "min_hvi", "count": "n_tracts"})
    .sort_values("avg_hvi", ascending=False)
    .round(2)
)

print("\n\nCITY-LEVEL HVI AVERAGES (top 15):")
print(city_stats.head(15).to_string())

# ── 4. Risk categorization ────────────────────────────────────────────────────

bins   = [0, 25, 35, 50, 65, 100]
labels = ["Very Low (<25)", "Low (25–35)", "Medium (35–50)",
          "High (50–65)", "Very High (>65)"]

df["risk_category"] = pd.cut(df["hvi"], bins=bins, labels=labels, right=True)

risk_counts = df["risk_category"].value_counts().sort_index()
print("\n\nRISK CATEGORY DISTRIBUTION:")
print(risk_counts.to_string())
print(f"\nTotal tracts analyzed: {len(df)}")

# ── 5. Visualization ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("LA County Heat Vulnerability Index (HVI) Analysis",
             fontsize=16, fontweight="bold", y=0.98)

# ── Plot 1: Choropleth map ────────────────────────────────────────────────────
ax1 = axes[0, 0]

city_avg = df.groupby("census_city")["hvi"].mean().reset_index()
city_avg.columns = ["name", "avg_hvi"]
merged = gdf.merge(city_avg, on="name", how="left")

norm  = Normalize(vmin=df["hvi"].min(), vmax=df["hvi"].max())
cmap  = plt.cm.YlOrRd

merged.plot(
    column="avg_hvi",
    cmap=cmap,
    linewidth=0.4,
    edgecolor="white",
    legend=False,
    missing_kwds={"color": "#d1d5db"},
    ax=ax1,
)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label("Avg HVI Score", fontsize=9)

ax1.set_title("Choropleth: Avg HVI by Neighborhood", fontsize=11, pad=8)
ax1.axis("off")

# ── Plot 2: HVI distribution histogram ───────────────────────────────────────
ax2 = axes[0, 1]

ax2.hist(df["hvi"], bins=30, color="#fb923c", edgecolor="white", linewidth=0.5)
ax2.axvline(df["hvi"].mean(),   color="#1d4ed8", linestyle="--",
            linewidth=1.5, label=f"Mean: {df['hvi'].mean():.1f}")
ax2.axvline(df["hvi"].median(), color="#15803d", linestyle="--",
            linewidth=1.5, label=f"Median: {df['hvi'].median():.1f}")

ax2.set_title("HVI Score Distribution", fontsize=11, pad=8)
ax2.set_xlabel("HVI Score")
ax2.set_ylabel("Number of Census Tracts")
ax2.legend(fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

# ── Plot 3: Top 15 cities by avg HVI (horizontal bar) ────────────────────────
ax3 = axes[1, 0]

top_cities = city_stats.head(15).reset_index()
colors = cmap(norm(top_cities["avg_hvi"].values))

bars = ax3.barh(top_cities["census_city"], top_cities["avg_hvi"],
                color=colors, edgecolor="white", linewidth=0.5)

ax3.set_title("Top 15 Cities by Average HVI", fontsize=11, pad=8)
ax3.set_xlabel("Average HVI Score")
ax3.invert_yaxis()
ax3.spines[["top", "right"]].set_visible(False)

for bar, val in zip(bars, top_cities["avg_hvi"]):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}", va="center", fontsize=8)

# ── Plot 4: Risk category pie chart ──────────────────────────────────────────
ax4 = axes[1, 1]

pie_colors = ["#4ade80", "#a3e635", "#fbbf24", "#f97316", "#dc2626"]
wedges, texts, autotexts = ax4.pie(
    risk_counts,
    labels=risk_counts.index,
    autopct="%1.1f%%",
    colors=pie_colors,
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1},
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(8)

ax4.set_title("Census Tracts by Risk Category", fontsize=11, pad=8)

plt.tight_layout()
plt.savefig("hvi_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → hvi_analysis.png")