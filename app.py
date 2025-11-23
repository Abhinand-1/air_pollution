import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import gdown

st.set_page_config(page_title="Kerala Pollution Dashboard (Kriging)", layout="wide")

# ---------------------------------------------------------------------
# 1) DOWNLOAD BIG DATASET FROM GOOGLE DRIVE (gdown supports any size)
# ---------------------------------------------------------------------
DATA_URL = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"
LOCAL_FILE = "kerala_pollution.csv"

# Kerala boundary uploaded earlier
BOUNDARY_PATH = "kerala_boundary.geojson"


@st.cache_data
def load_data():
    # Download only once (cached)
    if not os.path.exists(LOCAL_FILE):
        with st.spinner("Downloading large dataset from Google Drive..."):
            gdown.download(DATA_URL, LOCAL_FILE, quiet=False)

    # Load the CSV
    df = pd.read_csv(LOCAL_FILE)

    # Clean date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep valid rows
    df = df.dropna(subset=["date", "lat", "lon"])

    return df


# ---------------------------------------------------------------------
# 2) LOAD KERALA GEOJSON (LOCAL FILE)
# ---------------------------------------------------------------------
@st.cache_data
def load_kerala_polygon():
    boundary_path = "/mnt/data/state (1).geojson"   # correct path

    if not os.path.exists(boundary_path):
        st.error(f"Kerala boundary file not found at: {boundary_path}")
        st.stop()

    # Load the local geojson file
    with open(boundary_path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    features = geo["features"] if "features" in geo else [geo]

    polys = []
    for feat in features:
        geom = shape(feat["geometry"])
        if isinstance(geom, (Polygon, MultiPolygon)):
            polys.append(geom)

    from shapely.ops import unary_union
    return unary_union(polys)



def clip_points(df, polygon):
    pts = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    mask = [polygon.contains(p) for p in pts]
    return df[np.array(mask)].reset_index(drop=True)


# ---------------------------------------------------------------------
# 3) KRIGING
# ---------------------------------------------------------------------
def do_kriging(df_points, pollutant, grid_res=150):
    lons = df_points["lon"].values
    lats = df_points["lat"].values
    vals = df_points[pollutant].values

    pad_x = (lons.max() - lons.min()) * 0.02
    pad_y = (lats.max() - lats.min()) * 0.02

    gx = np.linspace(lons.min() - pad_x, lons.max() + pad_x, grid_res)
    gy = np.linspace(lats.min() - pad_y, lats.max() + pad_y, grid_res)

    OK = OrdinaryKriging(lons, lats, vals, variogram_model="spherical")

    z, ss = OK.execute("grid", gx, gy)

    return gx, gy, z


def mask_grid(gx, gy, z, polygon):
    xx, yy = np.meshgrid(gx, gy)
    pts = [Point(xy) for xy in zip(xx.ravel(), yy.ravel())]
    mask = np.array([polygon.contains(p) for p in pts])

    return pd.DataFrame({
        "lon": xx.ravel()[mask],
        "lat": yy.ravel()[mask],
        "value": z.ravel()[mask]
    })


# ---------------------------------------------------------------------
# 4) LOAD EVERYTHING
# ---------------------------------------------------------------------
df_all = load_data()
kerala_poly = load_kerala_polygon()

# ---------------------------------------------------------------------
# 5) SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.header("Controls")

pollutants = ["AOD", "NO2", "SO2", "CO", "O3"]
pollutants = [p for p in pollutants if p in df_all.columns]
pollutant = st.sidebar.selectbox("Pollutant", pollutants)

mode = st.sidebar.radio("Mode", [
    "Interactive Map",
    "Daily Animation",
    "Monthly Animation",
    "Heatmap",
    "Kriging Smooth Map"
])

sample = st.sidebar.slider("Sample Size", 500, 5000, 2000)

# Date filter
dmin, dmax = df_all["date"].min(), df_all["date"].max()
dr = st.sidebar.date_input("Date Range", [dmin, dmax])

try:
    start, end = map(pd.to_datetime, dr)
    df = df_all[(df_all["date"] >= start) & (df_all["date"] <= end)]
except:
    df = df_all.copy()

df = clip_points(df, kerala_poly)

if df.empty:
    st.error("No data points fall inside Kerala shape.")
    st.stop()

df_s = df.sample(min(sample, len(df)), random_state=42)

# ---------------------------------------------------------------------
# 6) MAIN TITLE
# ---------------------------------------------------------------------
st.title("🌏 Kerala Air Pollution Dashboard with Kriging")
st.write(f"Currently showing: **{pollutant}**")

# ---------------------------------------------------------------------
# 7) VISUAL MODES
# ---------------------------------------------------------------------

# ========== INTERACTIVE MAP ==========
if mode == "Interactive Map":
    fig = px.scatter_mapbox(
        df_s, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        zoom=7, height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ========== DAILY ANIMATION ==========
elif mode == "Daily Animation":
    df_s["frame"] = df_s["date"].dt.strftime("%Y-%m-%d")
    fig = px.scatter_mapbox(
        df_s, lat="lat", lon="lon",
        animation_frame="frame",
        color=pollutant, size=pollutant,
        zoom=7, height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ========== MONTHLY ANIMATION ==========
elif mode == "Monthly Animation":
    df_m = df.copy()
    df_m["month"] = df_m["date"].dt.to_period("M").astype(str)

    fig = px.scatter_mapbox(
        df_m, lat="lat", lon="lon",
        animation_frame="month",
        color=pollutant, size=pollutant,
        zoom=7, height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ========== HEATMAP ==========
elif mode == "Heatmap":
    fig = px.density_mapbox(
        df_s, lat="lat", lon="lon",
        z=pollutant, radius=25,
        zoom=7, height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ========== KRIGING ==========
elif mode == "Kriging Smooth Map":

    st.subheader("Kriging Interpolated Surface")

    if len(df_s) < 3:
        st.error("Need at least 3 points for kriging.")
        st.stop()

    with st.spinner("Performing kriging..."):
        gx, gy, z = do_kriging(df_s, pollutant)

    grid = mask_grid(gx, gy, z, kerala_poly)

    fig = px.density_mapbox(
        grid, lat="lat", lon="lon",
        z="value", radius=10,
        zoom=7, height=700,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")

    # Add original points
    fig.add_scattermapbox(
        lat=df_s["lat"], lon=df_s["lon"],
        mode="markers",
        marker=dict(size=5, color="black"),
        name="Data Points"
    )

    st.plotly_chart(fig, use_container_width=True)

