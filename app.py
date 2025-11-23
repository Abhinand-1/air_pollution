# app.py — Kerala Pollution Dashboard with Kriging (Shapely clipping)
import streamlit as st
import pandas as pd
import numpy as np
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import os

st.set_page_config(page_title="Kerala Pollution Dashboard (with Kriging)", layout="wide")

# ------------------------------------------------------------------
# CONFIG - update if you host files elsewhere
# Use the exact local paths you uploaded:
DATA_PATH = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"

BOUNDARY_PATH = "https://raw.githubusercontent.com/Abhinand-1/air_pollution/main/kerala_boundary.geojson"

# ------------------------------------------------------------------

# -------------------------
# 1) Load CSV
# -------------------------
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ensure lat/lon exist and numeric
    df = df.dropna(subset=["lat", "lon", "date"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    return df

# -------------------------
# 2) Load Kerala polygon using shapely + json (no geopandas)
# -------------------------
@st.cache_data
def load_kerala_polygon(geojson_path):
    # geojson_path can be local filesystem path (Streamlit Cloud will require the file in the repo)
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = []
    if isinstance(gj, dict) and "features" in gj:
        features = gj["features"]
    elif isinstance(gj, list):
        features = gj
    else:
        # single geometry object
        features = [gj]

    polys = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat
        if geom is None:
            continue
        shp = shape(geom)
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)

    if not polys:
        raise ValueError("No polygon features found in GeoJSON at: " + geojson_path)

    if len(polys) == 1:
        return polys[0]
    else:
        # unify
        from shapely.ops import unary_union
        return unary_union(polys)

# -------------------------
# 3) Clip points by polygon (shapely)
# -------------------------
def clip_points_to_polygon(df, polygon):
    # faster approach: vectorized containment via list comprehension
    points = [Point(xy) for xy in zip(df["lon"].values, df["lat"].values)]
    mask = [polygon.contains(pt) for pt in points]
    return df[np.array(mask)].reset_index(drop=True)

# -------------------------
# 4) Kriging function (Ordinary Kriging)
# -------------------------
def do_ordinary_kriging(df_points, pollutant, grid_res=150, variogram_model="spherical"):
    # Ensure arrays are 1d numpy
    lons = df_points["lon"].values.astype(float)
    lats = df_points["lat"].values.astype(float)
    vals = df_points[pollutant].values.astype(float)

    # Build grid bounds a little padded
    pad_lon = (lons.max() - lons.min()) * 0.02
    pad_lat = (lats.max() - lats.min()) * 0.02
    min_lon, max_lon = lons.min() - pad_lon, lons.max() + pad_lon
    min_lat, max_lat = lats.min() - pad_lat, lats.max() + pad_lat

    grid_lon = np.linspace(min_lon, max_lon, grid_res)
    grid_lat = np.linspace(min_lat, max_lat, grid_res)

    # Ordinary Kriging - PyKrige expects x (lon), y (lat)
    OK = OrdinaryKriging(
        lons, lats, vals,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    # execute: returns array shaped (len(grid_lat), len(grid_lon))
    z, ss = OK.execute("grid", grid_lon, grid_lat)  # z: (ny, nx)
    return grid_lon, grid_lat, z

# -------------------------
# 5) Mask grid by polygon, return flat DataFrame of points inside polygon
# -------------------------
def mask_grid_by_polygon(grid_lon, grid_lat, z, polygon):
    # create meshgrid
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)  # shapes (ny, nx)
    flat_lon = lon_mesh.ravel()
    flat_lat = lat_mesh.ravel()
    flat_z = z.ravel()

    pts = [Point(xy) for xy in zip(flat_lon, flat_lat)]
    mask = np.array([polygon.contains(p) for p in pts])

    if mask.sum() == 0:
        return pd.DataFrame(columns=["lon","lat","value"])

    inside_lon = flat_lon[mask]
    inside_lat = flat_lat[mask]
    inside_z = flat_z[mask]

    grid_df = pd.DataFrame({
        "lon": inside_lon,
        "lat": inside_lat,
        "value": inside_z
    })
    return grid_df

# -------------------------
# 6) Plotting helper
# -------------------------
def plot_density_from_grid_df(grid_df, title, center_lat, center_lon, zoom=7):
    fig = px.density_mapbox(
        grid_df,
        lat="lat",
        lon="lon",
        z="value",
        radius=12,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
        color_continuous_scale="Turbo",
        height=800,
    )
    fig.update_layout(mapbox_style="open-street-map", title=title)
    return fig

# -------------------------
# MAIN: Load data + polygon
# -------------------------
st.title("Kerala Pollution Dashboard — Kriging (Shapely clipping)")

with st.spinner("Loading data and boundary..."):
    # ensure files exist
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}. Upload df_final.csv to repository or update DATA_PATH.")
        st.stop()
    if not os.path.exists(BOUNDARY_PATH):
        st.error(f"Boundary GeoJSON not found at {BOUNDARY_PATH}. Upload Kerala GeoJSON to repository or update BOUNDARY_PATH.")
        st.stop()

    df_all = load_data(DATA_PATH)
    try:
        kerala_poly = load_kerala_polygon(BOUNDARY_PATH)
    except Exception as e:
        st.error(f"Failed to load Kerala GeoJSON: {e}")
        st.stop()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
pollutant_choices = [c for c in df_all.columns if c.upper() in ["AOD","NO2","SO2","CO","O3"]]
if not pollutant_choices:
    st.error("No pollutant columns found (AOD, NO2, SO2, CO, O3). Check your CSV.")
    st.stop()
pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_choices, index=0)
mode = st.sidebar.radio("View Mode", ["Interactive Map", "Daily Animation", "Monthly Animation", "Heatmap", "Kriging Smooth Map"])
sample_size = st.sidebar.slider("Sample Size (for plotting/kriging)", 500, 5000, 2000, step=100)
grid_res = st.sidebar.slider("Kriging grid resolution (side length)", 80, 300, 150, step=10)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical", "exponential", "gaussian"], index=0)

# date filter
date_min = df_all["date"].min()
date_max = df_all["date"].max()
date_range = st.sidebar.date_input("Date range", [date_min.date(), date_max.date()], min_value=date_min.date(), max_value=date_max.date())

try:
    start_dt = pd.to_datetime(pd.to_datetime(date_range[0]))
    end_dt = pd.to_datetime(pd.to_datetime(date_range[1])) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_all[(df_all["date"] >= start_dt) & (df_all["date"] <= end_dt)]
except Exception:
    df_filtered = df_all.copy()

# clip points to Kerala polygon (shapely)
df_clipped = clip_points_to_polygon(df_filtered, kerala_poly)
if df_clipped.empty:
    st.error("No data points found inside Kerala for the selected date range.")
    st.stop()

st.sidebar.write(f"Points inside Kerala: {len(df_clipped):,}")

# sample for plotting
df_sample_plot = df_clipped.sample(min(sample_size, len(df_clipped)), random_state=42)

# -------------------------
# Mode: Interactive Map
# -------------------------
if mode == "Interactive Map":
    st.subheader("Interactive pollution points")
    fig = px.scatter_mapbox(
        df_sample_plot, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        hover_data=["date", "lat", "lon", pollutant],
        zoom=7, height=750, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Mode: Daily Animation
# -------------------------
elif mode == "Daily Animation":
    st.subheader("Daily animation (sampled)")
    df_anim = df_sample_plot.copy()
    df_anim["frame"] = df_anim["date"].dt.strftime("%Y-%m-%d")
    fig = px.scatter_mapbox(
        df_anim,
        lat="lat", lon="lon", color=pollutant, size=pollutant,
        animation_frame="frame",
        hover_data=["date", pollutant],
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Mode: Monthly Animation (sorted correctly)
# -------------------------
elif mode == "Monthly Animation":
    st.subheader("Monthly mean animation")

    df_m = df_clipped.copy()
    df_m["year_month"] = df_m["date"].dt.to_period("M").astype(str)
    df_m["ym_dt"] = pd.to_datetime(df_m["year_month"], format="%Y-%m")
    df_month = (
        df_m.groupby(["ym_dt", "lat", "lon"])[pollutant].mean()
        .reset_index()
        .sort_values("ym_dt")
    )
    df_month["year_month"] = df_month["ym_dt"].dt.strftime("%Y-%m")
    df_month_s = df_month.sample(min(sample_size, len(df_month)), random_state=42)

    fig = px.scatter_mapbox(
        df_month_s, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        animation_frame="year_month",
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Mode: Heatmap
# -------------------------
elif mode == "Heatmap":
    st.subheader("Density heatmap")
    fig = px.density_mapbox(
        df_sample_plot, lat="lat", lon="lon",
        z=pollutant, radius=25,
        center=dict(lat=df_clipped["lat"].mean(), lon=df_clipped["lon"].mean()),
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Mode: Kriging Smooth Map
# -------------------------
elif mode == "Kriging Smooth Map":
    st.subheader("Kriging — Ordinary Kriging (spherical)")

    # Prepare points (drop na)
    df_pts = df_clipped.dropna(subset=[pollutant, "lat", "lon"]).copy()
    if len(df_pts) > sample_size:
        df_pts = df_pts.sample(sample_size, random_state=42)

    st.write(f"Using {len(df_pts)} points for kriging. Grid resolution: {grid_res}x{grid_res}. Variogram: {variogram_model}")

    # need at least 3 unique locations
    if df_pts["lon"].nunique() < 3 or df_pts["lat"].nunique() < 3:
        st.error("Not enough distinct point locations to perform kriging.")
    else:
        with st.spinner("Performing Ordinary Kriging (this may take a few seconds)..."):
            try:
                grid_lon, grid_lat, z = do_ordinary_kriging(df_pts, pollutant, grid_res=grid_res, variogram_model=variogram_model)
            except Exception as e:
                st.error(f"Kriging failed: {e}")
                st.stop()

        # mask grid to Kerala polygon
        grid_df = mask_grid_by_polygon(grid_lon, grid_lat, z, kerala_poly)
        if grid_df.empty:
            st.error("Kriged grid masked to polygon is empty. Try increasing grid resolution or sample points.")
        else:
            center_lat = df_clipped["lat"].mean()
            center_lon = df_clipped["lon"].mean()
            fig = plot_density_from_grid_df(grid_df, f"Kriged {pollutant}", center_lat, center_lon, zoom=7)

            # overlay sample points toggle
            if st.checkbox("Overlay sample points", value=True):
                fig.add_scattermapbox(
                    lat=df_pts["lat"],
                    lon=df_pts["lon"],
                    mode="markers",
                    marker=dict(size=6, color="black", opacity=0.6),
                    showlegend=False,
                    name="sample points"
                )

            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Notes: Kriging uses a variogram model and can be slow for many points. Tune sample size and grid resolution to balance speed vs quality.")
