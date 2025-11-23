# app.py - Kriging-enabled Kerala Pollution Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import os

st.set_page_config(page_title="Kerala Pollution Dashboard (Kriging)", layout="wide")

# -------------------------
# CONFIG - change paths if needed
# -------------------------
DATA_PATH = "/mnt/data/df_final.csv"               # <-- your uploaded CSV path
BOUNDARY_PATH = "/mnt/data/state (1).geojson"      # <-- your uploaded Kerala geojson path
DEFAULT_POLLUTANT = "NO2"

# -------------------------
# Utility: load data
# -------------------------
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "lat", "lon"])
    return df

# -------------------------
# Utility: load Kerala polygon (GeoJSON) using pure python+shapely
# -------------------------
@st.cache_data
def load_kerala_polygon(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # Accept FeatureCollection or single Feature
    features = gj.get("features", [gj]) if isinstance(gj, dict) else gj
    polys = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat
        if geom is None:
            continue
        shp = shape(geom)
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)
    if not polys:
        raise ValueError("No polygon found in GeoJSON")
    # unify into a single MultiPolygon or Polygon
    if len(polys) == 1:
        return polys[0]
    else:
        from shapely.ops import unary_union
        return unary_union(polys)

# -------------------------
# Utility: clip points to polygon (shapely)
# -------------------------
def clip_points_to_polygon(df, polygon):
    mask = df.apply(lambda r: polygon.contains(Point(r["lon"], r["lat"])), axis=1)
    return df[mask].reset_index(drop=True)

# -------------------------
# Kriging interpolation function
# -------------------------
def do_ordinary_kriging(df_points, pollutant, grid_res=150, variogram_model="spherical"):
    """
    df_points: DataFrame with columns ['lon','lat', pollutant]
    returns grid_lon, grid_lat, z (2D array of interpolated values)
    """
    # prepare arrays
    lons = df_points["lon"].values
    lats = df_points["lat"].values
    vals = df_points[pollutant].values

    # create grid
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    grid_lon = np.linspace(min_lon, max_lon, grid_res)
    grid_lat = np.linspace(min_lat, max_lat, grid_res)

    # PyKrige expects (x, y, z) as (long, lat, value)
    OK = OrdinaryKriging(
        lons, lats, vals,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    # execute on grid (note: PyKrige returns mesh with shape [len(grid_lat), len(grid_lon)])
    z, ss = OK.execute("grid", grid_lon, grid_lat)
    # z shape: (ny, nx) where ny = len(grid_lat), nx = len(grid_lon)
    # We'll return grid_lon (x axis), grid_lat (y axis), and z.T shaped accordingly for flattening
    return grid_lon, grid_lat, z  # z indexed as [lat_index, lon_index]

# -------------------------
# Mask grid by polygon
# -------------------------
def mask_grid_by_polygon(grid_lon, grid_lat, z, polygon):
    long, latt = np.meshgrid(grid_lon, grid_lat)
    flat_lon = long.ravel()
    flat_lat = latt.ravel()
    flat_z = z.ravel()

    pts = [Point(xy) for xy in zip(flat_lon, flat_lat)]
    mask = np.array([polygon.contains(pt) for pt in pts])

    # Keep only points inside polygon
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
# Plotting helper: plot kriged grid as density_mapbox
# -------------------------
def plot_kriging_grid(grid_df, pollutant_name, center_lat, center_lon, zoom=7):
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
        labels={"value": pollutant_name}
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig

# -------------------------
# MAIN: UI + flow
# -------------------------
st.title("Kerala Pollution Dashboard — Kriging Interpolation")

# Load data and polygon
with st.spinner("Loading data..."):
    df_all = load_data(DATA_PATH)
    try:
        kerala_poly = load_kerala_polygon(BOUNDARY_PATH)
    except Exception as e:
        st.error(f"Failed to load Kerala boundary GeoJSON: {e}")
        st.stop()

# sidebar controls
st.sidebar.header("Options")
pollutant = st.sidebar.selectbox("Pollutant", sorted([c for c in df_all.columns if c.lower() in ["aod","no2","so2","co","o3"] or c in ["AOD","NO2","SO2","CO","O3"]]), index=0)
mode = st.sidebar.radio("Mode", ["Interactive points", "Kriging Smooth Map"])
sample_n = st.sidebar.slider("Max sample points (for kriging speed)", 500, 5000, 2000, step=100)
grid_res = st.sidebar.slider("Grid resolution (side length)", 80, 300, 150, step=10)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical", "exponential", "gaussian"], index=0)

# clip points to Kerala
df_clipped = clip_points_to_polygon(df_all, kerala_poly)
if df_clipped.empty:
    st.error("No points found inside Kerala polygon after clipping.")
    st.stop()

st.sidebar.write(f"Points inside Kerala: {len(df_clipped):,}")

# date filter
min_date = df_clipped["date"].min()
max_date = df_clipped["date"].max()
date_range = st.sidebar.date_input("Date range", value=[min_date.date(), max_date.date()], min_value=min_date.date(), max_value=max_date.date())
try:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_clipped = df_clipped[(df_clipped["date"] >= start_dt) & (df_clipped["date"] <= end_dt)]
except Exception:
    pass

if df_clipped.empty:
    st.warning("No data for selected date range.")
    st.stop()

# Simple interactive points mode
if mode == "Interactive points":
    st.subheader("Interactive sampled points")
    display_sample = df_clipped.sample(min(5000, len(df_clipped)), random_state=42)
    fig = px.scatter_mapbox(
        display_sample,
        lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        hover_data=["date", pollutant],
        zoom=7,
        height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# Kriging mode
else:
    st.subheader("Ordinary Kriging (smooth interpolated map)")

    # sample points for kriging
    df_pts = df_clipped.dropna(subset=[pollutant, "lat", "lon"])
    if len(df_pts) > sample_n:
        df_pts = df_pts.sample(sample_n, random_state=42)

    st.write(f"Using {len(df_pts)} sample points for kriging.")

    # Prepare arrays
    # check for enough distinct points
    if df_pts["lon"].nunique() < 3 or df_pts["lat"].nunique() < 3:
        st.error("Not enough spatial variation to perform kriging. Need more distinct point locations.")
        st.stop()

    # Do kriging
    with st.spinner("Performing kriging (this may take a few seconds)..."):
        try:
            grid_lon, grid_lat, z = do_ordinary_kriging(df_pts, pollutant, grid_res=grid_res, variogram_model=variogram_model)
        except Exception as e:
            st.error(f"Kriging failed: {e}\nMake sure pykrige, scipy and numpy are installed.")
            st.stop()

    # mask grid to polygon and plot
    grid_df = mask_grid_by_polygon(grid_lon, grid_lat, z, kerala_poly)
    if grid_df.empty:
        st.error("Kriged grid masked to polygon is empty. Try increasing grid resolution or sample points.")
        st.stop()

    center_lat = df_clipped["lat"].mean()
    center_lon = df_clipped["lon"].mean()

    fig = plot_kriging_grid(grid_df, pollutant, center_lat, center_lon, zoom=7)
    st.plotly_chart(fig, use_container_width=True)

    # optional: overlay original points
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
st.write("Notes: Kriging uses a variogram model and can be slow for many points. Use sample size and grid resolution to balance speed vs quality.")
