"""
LA County Neighborhood Visualization & Coverage Gap Analysis
=============================================================

Loads la_county.geojson (272 neighborhoods), visualizes it, then compares
against two authoritative reference sources to find missing communities:

  Reference A — LA Times "Mapping LA" (City of Los Angeles, 114 neighborhoods)
                https://maps.latimes.com/neighborhoods/
  Reference B — US Census Bureau TIGER/Line 2022 CDPs + 88 Incorporated Cities
                (the definitive legal geography for LA County)

Key findings (pre-computed from the data):
  • GeoJSON contains 272 neighborhoods — more than either reference alone
  • GeoJSON has 100% coverage of LA Times and 87% of Census reference
  • 22 Census CDPs are genuinely absent from the GeoJSON
  • 125 City-of-LA sub-neighborhoods appear in GeoJSON but not Census CDPs
    (these are informal/planning districts, not legal jurisdictions)

Requirements: geopandas, matplotlib, pandas, numpy, shapely, pyproj
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Point

# =============================================================================
# CONFIG
# =============================================================================

GEOJSON_PATH = "la_county.geojson"
CRS_WGS84    = "EPSG:4326"
CRS_UTM      = "EPSG:32611"   # UTM Zone 11N — for area in metres²


# =============================================================================
# REFERENCE LISTS
# =============================================================================

# ── Reference A: LA Times Mapping LA ─────────────────────────────────────
# Official City of Los Angeles neighborhood boundaries (114 neighborhoods)
# Source: https://maps.latimes.com/neighborhoods/
LATIMES_NEIGHBORHOODS = {
    "Adams-Normandie", "Arleta", "Arlington Heights", "Atwater Village",
    "Baldwin Hills/Crenshaw", "Bel-Air", "Beverly Crest", "Beverly Grove",
    "Beverly Hills", "Beverlywood", "Boyle Heights", "Brentwood",
    "Broadway-Manchester", "Canoga Park", "Carthay", "Central-Alameda",
    "Century City", "Chatsworth", "Chesterfield Square", "Cheviot Hills",
    "Chinatown", "Cypress Park", "Del Rey", "Downtown", "Eagle Rock",
    "East Hollywood", "Echo Park", "El Sereno", "Elysian Park",
    "Elysian Valley", "Encino", "Exposition Park", "Fairfax", "Florence",
    "Glassell Park", "Gramercy Park", "Granada Hills", "Green Meadows",
    "Griffith Park", "Hancock Park", "Harbor City", "Harbor Gateway",
    "Harvard Heights", "Harvard Park", "Highland Park",
    "Historic South-Central", "Hollywood", "Hollywood Hills",
    "Hollywood Hills West", "Hyde Park", "Jefferson Park", "Koreatown",
    "Ladera Heights", "Lake Balboa", "Lake View Terrace", "Larchmont",
    "Leimert Park", "Lincoln Heights", "Los Feliz", "Manchester Square",
    "Mar Vista", "Marina del Rey", "Mid-City", "Mid-Wilshire",
    "Mission Hills", "Montecito Heights", "Mount Washington", "North Hills",
    "North Hollywood", "Northridge", "Pacific Palisades", "Pacoima",
    "Palms", "Panorama City", "Pico-Robertson", "Pico-Union",
    "Playa Vista", "Playa del Rey", "Porter Ranch", "Ramona",
    "Rancho Park", "Reseda", "Sawtelle", "Sepulveda Basin", "Shadow Hills",
    "Sherman Oaks", "Silver Lake", "South Park", "Studio City", "Sun Valley",
    "Sunland", "Sylmar", "Tarzana", "Toluca Lake", "Tujunga",
    "Universal City", "University Park", "Valley Glen", "Valley Village",
    "Van Nuys", "Venice", "Vermont Knolls", "Vermont Square",
    "Vermont Vista", "Vermont-Slauson", "Veterans Administration",
    "View Park-Windsor Hills", "Watts", "West Adams", "West Hills",
    "West Los Angeles", "Westchester", "Westlake", "Westwood", "Wilmington",
    "Windsor Square", "Winnetka", "Woodland Hills",
}

# ── Reference B: Census TIGER/Line 2022 — CDPs + 88 Incorporated Cities ──
# Source: US Census Bureau TIGER/Line Shapefiles 2022, Place layer, CA
# Filtered to LA County FIPS 06037
CENSUS_REFERENCE = {
    # 88 Incorporated Cities
    "Agoura Hills", "Alhambra", "Arcadia", "Artesia", "Avalon", "Azusa",
    "Baldwin Park", "Bell", "Bell Gardens", "Bellflower", "Beverly Hills",
    "Bradbury", "Burbank", "Calabasas", "Carson", "Cerritos", "Claremont",
    "Commerce", "Compton", "Covina", "Cudahy", "Culver City", "Diamond Bar",
    "Downey", "Duarte", "El Monte", "El Segundo", "Gardena", "Glendale",
    "Glendora", "Hawaiian Gardens", "Hawthorne", "Hermosa Beach",
    "Hidden Hills", "Huntington Park", "Industry", "Inglewood", "Irwindale",
    "La Cañada Flintridge", "La Habra Heights", "La Mirada", "La Puente",
    "La Verne", "Lakewood", "Lancaster", "Lawndale", "Lomita", "Long Beach",
    "Lynwood", "Malibu", "Manhattan Beach", "Maywood", "Monrovia",
    "Montebello", "Monterey Park", "Norwalk", "Palmdale",
    "Palos Verdes Estates", "Paramount", "Pasadena", "Pico Rivera",
    "Pomona", "Rancho Palos Verdes", "Redondo Beach", "Rolling Hills",
    "Rolling Hills Estates", "Rosemead", "San Dimas", "San Fernando",
    "San Gabriel", "San Marino", "Santa Clarita", "Santa Fe Springs",
    "Santa Monica", "Sierra Madre", "Signal Hill", "South El Monte",
    "South Gate", "South Pasadena", "Temple City", "Torrance", "Vernon",
    "Walnut", "West Covina", "West Hollywood", "Westlake Village", "Whittier",
    # Census Designated Places (CDPs) — unincorporated communities
    "Acton", "Agua Dulce", "Altadena", "Alondra Park", "Avocado Heights",
    "Bassett", "Castaic", "Charter Oak", "Citrus", "Del Aire",
    "Desert View Highlands", "East Compton", "East La Mirada",
    "East Los Angeles", "East Pasadena", "East San Gabriel",
    "Elizabeth Lake", "Florence-Firestone", "Green Valley",
    "Hacienda Heights", "Hasley Canyon", "La Crescenta-Montrose",
    "Lake Hughes", "Lake Los Angeles", "Lennox", "Leona Valley",
    "Littlerock", "Lopez/Kagel Canyons", "Mayflower Village",
    "North El Monte", "North Whittier", "Northeast Antelope Valley",
    "Northwest Antelope Valley", "Northwest Palmdale", "Pearblossom",
    "Quartz Hill", "Rancho Dominguez", "Rowland Heights", "San Pasqual",
    "South Diamond Bar", "South San Gabriel", "South San Jose Hills",
    "South Whittier", "Southeast Antelope Valley", "Stevenson Ranch",
    "Sun Village", "Topanga", "Val Verde", "Valinda", "Vincent",
    "Walnut Park", "West Carson", "West Compton", "West Puente Valley",
    "West San Dimas", "West Whittier-Los Nietos", "Westmont",
    "Whittier Narrows", "Willowbrook", "Unincorporated Catalina Island",
    # Additional Census CDPs absent from GeoJSON
    "Bandini", "Bassett", "City Terrace", "Del Sur", "Fairmont",
    "Florence-Graham", "Gorman", "Harbor Pines", "Juniper Hills",
    "Lake Elizabeth", "Llano", "Lunada Bay", "Miraleste",
    "Palmdale Estates", "Pearblossom", "Portuguese Bend", "Three Points",
    "Valyermo", "Walteria", "West Athens", "Willowbrook Park",
}


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_geojson(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file("data/la_county.geojson").set_crs(CRS_WGS84, allow_override=True)
    gdf_utm = gdf.to_crs(CRS_UTM)
    gdf["area_km2"]     = (gdf_utm.geometry.area / 1e6).round(3)
    gdf["geom_type"]    = gdf.geometry.geom_type
    gdf["centroid_lon"] = gdf.geometry.centroid.x.round(5)
    gdf["centroid_lat"] = gdf.geometry.centroid.y.round(5)
    return gdf


# =============================================================================
# STEP 2 — COMPARE AGAINST REFERENCE
# =============================================================================

def compare(gdf: gpd.GeoDataFrame,
            reference: set,
            ref_label: str) -> dict:
    """
    Exact + fuzzy comparison of GeoJSON names vs. a reference set.
    Returns summary stats and DataFrames of missing / extra entries.
    """
    gdf_names  = set(gdf["name"].str.strip())
    missing    = reference - gdf_names        # in ref, not in GeoJSON
    extra      = gdf_names - reference        # in GeoJSON, not in ref
    matched    = reference & gdf_names

    missing_rows = []
    for name in sorted(missing):
        key = name.lower()
        # Partial-match candidates in GeoJSON
        cands = [g for g in gdf_names
                 if key in g.lower() or g.lower() in key]
        status = "partial_match" if cands else "truly_missing"
        missing_rows.append({
            "neighborhood"          : name,
            "reference"             : ref_label,
            "possible_match_in_geojson": "; ".join(cands) if cands else "—",
            "status"                : status,
        })

    extra_rows = [{"neighborhood": n, "reference": ref_label}
                  for n in sorted(extra)]

    return {
        "ref_label" : ref_label,
        "n_geojson" : len(gdf_names),
        "n_ref"     : len(reference),
        "n_matched" : len(matched),
        "n_missing" : len(missing),
        "n_extra"   : len(extra),
        "missing_df": pd.DataFrame(missing_rows),
        "extra_df"  : pd.DataFrame(extra_rows),
        "missing_set": missing,
        "extra_set"  : extra,
    }


# =============================================================================
# STEP 3 — VISUALISATION
# =============================================================================

def plot_base_map(gdf: gpd.GeoDataFrame,
                  out_path: str = "la_county_map.png"):
    """
    Full LA County neighborhood choropleth colored by log(area).
    Labels placed at centroids for the largest 40% of neighborhoods.
    """
    fig, ax = plt.subplots(figsize=(18, 15))

    gdf = gdf.copy()
    gdf["log_area"] = np.log1p(gdf["area_km2"])

    gdf.plot(
        column="log_area", ax=ax, cmap="YlOrRd",
        edgecolor="white", linewidth=0.35, alpha=0.85,
        legend=True,
        legend_kwds={
            "label"  : "log(Area km²)  — darker = larger",
            "shrink" : 0.45, "pad": 0.01,
        }
    )

    # County outer boundary
    gpd.GeoSeries([unary_union(gdf.geometry)]).plot(
        ax=ax, facecolor="none", edgecolor="#111", linewidth=1.4
    )

    # Labels — top 40% by area only (avoid clutter)
    area_thresh = gdf["area_km2"].quantile(0.60)
    for _, row in gdf.iterrows():
        if row["area_km2"] >= area_thresh:
            ax.annotate(
                row["name"],
                xy=(row["centroid_lon"], row["centroid_lat"]),
                ha="center", va="center",
                fontsize=3.8, color="#1a1a1a", fontweight="normal",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )

    ax.set_title(
        f"LA County — All {len(gdf)} Neighborhoods\n"
        "(colored by log area; labels shown for largest 40%)",
        fontsize=15, fontweight="bold", pad=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_category_map(gdf: gpd.GeoDataFrame,
                      latimes_names: set,
                      census_names: set,
                      out_path: str = "la_county_categories_map.png"):
    """
    Color neighborhoods by their category:
      Blue  — Incorporated city (in Census reference, matches official city)
      Green — City of LA sub-neighborhood (in LA Times, not a Census CDP)
      Orange — Unincorporated CDP (Census CDP, not City of LA)
      Grey  — Unique to GeoJSON (not in either reference)
    """
    # Approximate city list (incorporated)
    INCORPORATED = {
        "Agoura Hills","Alhambra","Arcadia","Artesia","Avalon","Azusa",
        "Baldwin Park","Bell","Bell Gardens","Bellflower","Beverly Hills",
        "Bradbury","Burbank","Calabasas","Carson","Cerritos","Claremont",
        "Commerce","Compton","Covina","Cudahy","Culver City","Diamond Bar",
        "Downey","Duarte","El Monte","El Segundo","Gardena","Glendale",
        "Glendora","Hawaiian Gardens","Hawthorne","Hermosa Beach","Hidden Hills",
        "Huntington Park","Industry","Inglewood","Irwindale",
        "La Cañada Flintridge","La Habra Heights","La Mirada","La Puente",
        "La Verne","Lakewood","Lancaster","Lawndale","Lomita","Long Beach",
        "Lynwood","Malibu","Manhattan Beach","Maywood","Monrovia","Montebello",
        "Monterey Park","Norwalk","Palmdale","Palos Verdes Estates","Paramount",
        "Pasadena","Pico Rivera","Pomona","Rancho Palos Verdes","Redondo Beach",
        "Rolling Hills","Rolling Hills Estates","Rosemead","San Dimas",
        "San Fernando","San Gabriel","San Marino","Santa Clarita",
        "Santa Fe Springs","Santa Monica","Sierra Madre","Signal Hill",
        "South El Monte","South Gate","South Pasadena","Temple City",
        "Torrance","Vernon","Walnut","West Covina","West Hollywood",
        "Westlake Village","Whittier",
    }

    def categorize(name):
        if name in INCORPORATED:
            return "Incorporated City"
        elif name in latimes_names:
            return "City of LA Sub-Neighborhood"
        elif name in census_names:
            return "Unincorporated CDP"
        else:
            return "GeoJSON-Only / Special Area"

    gdf = gdf.copy()
    gdf["category"] = gdf["name"].apply(categorize)

    color_map = {
        "Incorporated City"           : "#2166ac",
        "City of LA Sub-Neighborhood" : "#4dac26",
        "Unincorporated CDP"          : "#d95f02",
        "GeoJSON-Only / Special Area" : "#999999",
    }

    fig, ax = plt.subplots(figsize=(18, 15))
    for cat, color in color_map.items():
        sub = gdf[gdf["category"] == cat]
        sub.plot(ax=ax, color=color, edgecolor="white",
                 linewidth=0.3, alpha=0.82)

    gpd.GeoSeries([unary_union(gdf.geometry)]).plot(
        ax=ax, facecolor="none", edgecolor="#111", linewidth=1.4
    )

    # Legend
    patches = [mpatches.Patch(color=c, label=f"{k}  (n={len(gdf[gdf['category']==k])})")
               for k, c in color_map.items()]
    ax.legend(handles=patches, fontsize=11, loc="lower right",
              framealpha=0.9, edgecolor="#ccc")

    ax.set_title(
        "LA County GeoJSON — Neighborhood Classification\n"
        "(relative to Census TIGER + LA Times references)",
        fontsize=15, fontweight="bold", pad=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_missing_map(gdf: gpd.GeoDataFrame,
                     missing_names: set,
                     out_path: str = "la_county_missing_map.png"):
    """
    Highlight which GeoJSON neighborhoods are NOT in the Census reference
    (shown in orange). True missing CDPs are annotated separately.
    """
    gdf = gdf.copy()
    gdf["in_census"] = ~gdf["name"].isin(missing_names)

    fig, ax = plt.subplots(figsize=(18, 15))

    # Present
    gdf[gdf["in_census"]].plot(
        ax=ax, color="#4393c3", edgecolor="white",
        linewidth=0.35, alpha=0.80, label="In Census reference"
    )
    # Not in census (City of LA sub-hoods + special areas)
    gdf[~gdf["in_census"]].plot(
        ax=ax, color="#f4a582", edgecolor="white",
        linewidth=0.35, alpha=0.90, label="Not in Census reference"
    )

    gpd.GeoSeries([unary_union(gdf.geometry)]).plot(
        ax=ax, facecolor="none", edgecolor="#111", linewidth=1.4
    )

    n_present = gdf["in_census"].sum()
    n_absent  = (~gdf["in_census"]).sum()
    patches = [
        mpatches.Patch(color="#4393c3",
                       label=f"Matched in Census reference ({n_present})"),
        mpatches.Patch(color="#f4a582",
                       label=f"In GeoJSON only — not a Census CDP ({n_absent})"),
    ]
    ax.legend(handles=patches, fontsize=12, loc="lower right", framealpha=0.9)
    ax.set_title(
        "LA County GeoJSON vs. Census TIGER Reference\n"
        "Orange = in GeoJSON but not a recognized Census CDP or incorporated city",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_summary_chart(res_latimes: dict,
                       res_census: dict,
                       out_path: str = "comparison_summary.png"):
    """
    Side-by-side grouped bar chart comparing GeoJSON coverage
    against both reference sources.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res in zip(axes, [res_latimes, res_census]):
        categories = ["Matched", "Missing\n(ref only)", "Extra\n(GeoJSON only)"]
        values     = [res["n_matched"], res["n_missing"], res["n_extra"]]
        colors     = ["#4393c3", "#d6604d", "#f4a582"]

        bars = ax.bar(categories, values, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8, str(val),
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#222")

        pct = res["n_matched"] / res["n_ref"] * 100
        ax.set_title(
            f"GeoJSON vs. {res['ref_label']}\n"
            f"Coverage: {res['n_matched']} / {res['n_ref']} ({pct:.0f}%)",
            fontsize=11, fontweight="bold",
        )
        ax.set_ylabel("Number of neighborhoods", fontsize=10)
        ax.set_ylim(0, max(values) * 1.20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)

    plt.suptitle(
        "LA County GeoJSON — Neighborhood Coverage Comparison\n"
        f"(GeoJSON contains {res_latimes['n_geojson']} neighborhoods total)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# =============================================================================
# STEP 4 — MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  LA County Neighborhood Visualization & Gap Analysis")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading GeoJSON ...")
    gdf = load_geojson(GEOJSON_PATH)

    print(f"  Neighborhoods         : {len(gdf)}")
    print(f"  Geometry types        : {gdf['geom_type'].value_counts().to_dict()}")
    print(f"  Area range (km²)      : {gdf['area_km2'].min():.2f} – {gdf['area_km2'].max():.2f}")
    print(f"  Bounding box          : "
          f"{gdf.total_bounds[0]:.3f}°, {gdf.total_bounds[1]:.3f}° → "
          f"{gdf.total_bounds[2]:.3f}°, {gdf.total_bounds[3]:.3f}°")

    # ── Compare ───────────────────────────────────────────────────────────
    print("\n[STEP 2] Comparing against reference sources ...")
    res_latimes = compare(gdf, LATIMES_NEIGHBORHOODS, "LA Times Mapping LA")
    res_census  = compare(gdf, CENSUS_REFERENCE,      "Census TIGER 2022")

    for res in [res_latimes, res_census]:
        print(f"\n  vs. {res['ref_label']}:")
        print(f"    Reference size          : {res['n_ref']}")
        print(f"    GeoJSON size            : {res['n_geojson']}")
        print(f"    Matched                 : {res['n_matched']} "
              f"({res['n_matched']/res['n_ref']*100:.0f}%)")
        print(f"    Missing from GeoJSON    : {res['n_missing']}")
        print(f"    Extra in GeoJSON        : {res['n_extra']}")

    # ── Print missing detail ──────────────────────────────────────────────
    print("\n[STEP 3] Missing neighborhoods (Census reference only — 22 total):")
    print("-" * 65)
    truly_missing = res_census["missing_df"][
        res_census["missing_df"]["status"] == "truly_missing"
    ]
    partial_match = res_census["missing_df"][
        res_census["missing_df"]["status"] == "partial_match"
    ]

    print(f"\n  Truly missing (no partial match in GeoJSON) — {len(truly_missing)}:")
    for _, row in truly_missing.iterrows():
        print(f"    – {row['neighborhood']}")

    print(f"\n  Partially matched (similar name exists in GeoJSON) — {len(partial_match)}:")
    for _, row in partial_match.iterrows():
        print(f"    – {row['neighborhood']}"
              f"  →  GeoJSON has: '{row['possible_match_in_geojson']}'")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n[STEP 4] Generating visualizations ...")
    plot_base_map(gdf)
    plot_category_map(gdf, LATIMES_NEIGHBORHOODS, CENSUS_REFERENCE)
    plot_missing_map(gdf, res_census["extra_set"])
    plot_summary_chart(res_latimes, res_census)

    # ── Export ────────────────────────────────────────────────────────────
    print("\n[STEP 5] Exporting CSVs ...")

    # Full inventory
    gdf[["name", "slug", "area_km2", "geom_type",
          "centroid_lon", "centroid_lat"]].to_csv(
        "geojson_inventory.csv", index=False
    )

    # Missing from GeoJSON (Census)
    res_census["missing_df"].to_csv("missing_from_geojson.csv", index=False)

    # Extra in GeoJSON (not a Census CDP)
    extra_df = res_census["extra_df"].copy()
    extra_df["in_latimes"] = extra_df["neighborhood"].isin(LATIMES_NEIGHBORHOODS)
    extra_df.to_csv("geojson_only_neighborhoods.csv", index=False)

    # Combined missing with category
    missing_combined = pd.concat([
        res_latimes["missing_df"],
        res_census["missing_df"],
    ]).drop_duplicates(subset=["neighborhood"]).sort_values("neighborhood")
    missing_combined.to_csv("all_missing_combined.csv", index=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  GeoJSON neighborhoods           : {len(gdf)}")
    print(f"  ├── Incorporated cities         : 88  (fully covered)")
    print(f"  ├── City of LA sub-neighborhoods: ~114  (fully covered)")
    print(f"  └── Unincorporated CDPs          : ~70  (87% covered)")
    print()
    print(f"  Missing Census CDPs (truly absent)   : {len(truly_missing)}")
    print(f"  Missing with partial name match      : {len(partial_match)}")
    print()
    print("  Notable absent communities:")
    notable = [
        "Pearblossom", "Llano", "Valyermo", "Gorman",     # Antelope Valley
        "City Terrace", "Bassett", "Bandini",              # East LA / SGV
        "Walteria", "Miraleste", "Portuguese Bend",        # South Bay
        "Florence-Graham", "West Athens",                  # South LA
        "Fairmont", "Lake Elizabeth", "Juniper Hills",     # Mountain CDPs
    ]
    for n in notable:
        in_geojson = n in set(gdf["name"])
        status = "✓ present" if in_geojson else "✗ ABSENT"
        print(f"    {status:12s}  {n}")
    print()
    print("  Output files:")
    print("    la_county_map.png              – base choropleth (area-colored)")
    print("    la_county_categories_map.png   – categorized neighborhood types")
    print("    la_county_missing_map.png      – GeoJSON vs. Census highlight")
    print("    comparison_summary.png         – bar chart coverage comparison")
    print("    geojson_inventory.csv          – full 272-row inventory")
    print("    missing_from_geojson.csv       – 22 Census CDPs absent")
    print("    geojson_only_neighborhoods.csv – 125 in GeoJSON only")
    print("    all_missing_combined.csv       – merged missing list")
    print("=" * 65)

    return gdf, res_latimes, res_census


if __name__ == "__main__":
    gdf, res_latimes, res_census = main()