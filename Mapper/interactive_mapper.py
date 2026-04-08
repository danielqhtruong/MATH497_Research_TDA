import pandas as pd
import folium
from folium.plugins import Search
import json
import numpy as np

# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/HVI_LACounty.csv")
df.columns = df.columns.str.strip()
df["census_tract"] = df["census_tract"].astype(str)
df["hvi"] = pd.to_numeric(df["hvi"], errors="coerce")
df = df.dropna(subset=["hvi"])

with open("data/la_county.geojson") as f:
    geo = json.load(f)

# ── 2. City-level aggregation ─────────────────────────────────────────────────

city_stats = (
    df.groupby("census_city")["hvi"]
    .agg(mean="mean", median="median", max="max", min="min", count="count")
    .round(2)
    .reset_index()
)

# Inject stats into GeoJSON properties so Folium tooltip can read them
city_lookup = city_stats.set_index("census_city").to_dict(orient="index")

for feature in geo["features"]:
    name = feature["properties"].get("name", "")
    stats = city_lookup.get(name, {})
    feature["properties"]["avg_hvi"]   = round(stats.get("mean",   0), 1)
    feature["properties"]["med_hvi"]   = round(stats.get("median", 0), 1)
    feature["properties"]["max_hvi"]   = round(stats.get("max",    0), 1)
    feature["properties"]["min_hvi"]   = round(stats.get("min",    0), 1)
    feature["properties"]["n_tracts"]  = int(stats.get("count",    0))
    feature["properties"]["has_data"]  = name in city_lookup

# ── 3. Color scale ────────────────────────────────────────────────────────────

def hvi_color(value):
    """Map HVI score to a hex color (yellow → orange → red)."""
    if not value or value == 0:
        return "#d1d5db"          # gray — no data
    if value < 25:
        return "#fef9c3"          # very light yellow
    elif value < 35:
        return "#fde68a"          # yellow
    elif value < 50:
        return "#fb923c"          # orange
    elif value < 65:
        return "#ef4444"          # red
    else:
        return "#7f1d1d"          # dark red

def style_function(feature):
    avg = feature["properties"].get("avg_hvi", 0)
    return {
        "fillColor":   hvi_color(avg),
        "color":       "white",
        "weight":      0.8,
        "fillOpacity": 0.75 if avg > 0 else 0.3,
    }

def highlight_function(feature):
    return {
        "fillColor":   hvi_color(feature["properties"].get("avg_hvi", 0)),
        "color":       "#1d4ed8",   # blue outline on hover
        "weight":      2.5,
        "fillOpacity": 0.95,
    }

# ── 4. Build map ──────────────────────────────────────────────────────────────

m = folium.Map(
    location=[34.05, -118.25],
    zoom_start=10,
    tiles="CartoDB positron",    # clean light basemap
    control_scale=True,
)

# ── 5. Choropleth layer ───────────────────────────────────────────────────────

tooltip = folium.GeoJsonTooltip(
    fields=["name", "avg_hvi", "med_hvi", "max_hvi", "min_hvi", "n_tracts"],
    aliases=["Neighborhood", "Avg HVI", "Median HVI", "Max HVI", "Min HVI", "Census tracts"],
    localize=True,
    sticky=True,                 # tooltip follows cursor
    labels=True,
    style="""
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
        font-family: sans-serif;
    """,
)

popup = folium.GeoJsonPopup(
    fields=["name", "avg_hvi", "med_hvi", "max_hvi", "min_hvi", "n_tracts"],
    aliases=["<b>Neighborhood</b>", "Avg HVI", "Median HVI",
             "Max HVI", "Min HVI", "Census tracts"],
    localize=True,
    labels=True,
    style="font-family: sans-serif; font-size: 13px;",
)

geojson_layer = folium.GeoJson(
    geo,
    name="HVI by Neighborhood",
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=tooltip,
    popup=popup,
).add_to(m)

# ── 6. Legend ─────────────────────────────────────────────────────────────────

legend_html = """
<div style="
    position: fixed; bottom: 40px; left: 40px; z-index: 1000;
    background: white; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 14px 18px;
    font-family: sans-serif; font-size: 13px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
">
  <div style="font-weight:600; margin-bottom:8px;">HVI Score</div>
  <div><span style="background:#fef9c3;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Very Low (&lt; 25)</div>
  <div><span style="background:#fde68a;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Low (25–35)</div>
  <div><span style="background:#fb923c;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Medium (35–50)</div>
  <div><span style="background:#ef4444;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>High (50–65)</div>
  <div><span style="background:#7f1d1d;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Very High (&gt; 65)</div>
  <div><span style="background:#d1d5db;display:inline-block;width:14px;height:14px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>No data</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── 7. Layer control + save ───────────────────────────────────────────────────

folium.LayerControl().add_to(m)

output_path = "hvi_interactive_map.html"
m.save(output_path)
print(f"Saved → {output_path}")
print("Open the file in any browser — no server needed.")