# app.py - Final: gdown + local path friendly + Kriging + Shapely clipping
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pykrige.ok import OrdinaryKriging
import plotly.express as px

st.set_page_config(page_title="Kerala Pollution Dashboard (Kriging)", layout="wide")

# ----------------------------
# CONFIG - use the local uploaded path (will be transformed by environment)
# If you uploaded a file, the path in conversation history is used here:
# (developer note: environment will transform local path to accessible URL)
DATA_URL = "/mnt/data/df_final.csv"   # <- local uploaded file path from your history
# If you prefer Google Drive direct link, set DATA_URL to:
# "https://drive.google.com/uc?id=YOUR_FILE_ID"
LOCAL_FILENAME = "kerala_pollution.csv"

BOUNDARY_PATH = "/mnt/data/state (1).geojson"  # uploaded geojson path

# ----------------------------
# Helpers: load CSV (local path preferred, otherwise download via gdown)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(data_url=DATA_URL, local_filename=LOCAL_FILENAME):
    # If data_url looks like a local path and file exists, read directly
    try:
        if data_url.startswith("/mnt/") or os.path.exists(data_url):
            path = data_url
            # If environment translated the path to a URL, pandas can read it directly.
            df = pd.read_csv(path)
        else:
            # assume Google Drive style or remote url — try gdown.download to local file
            try:
                import gdown
            except Exception as e:
                raise RuntimeError("gdown not installed. Add gdown to requirements.txt") from e
            # Accept both uc?id=... or /file/d/... links
            # If user passed uc? link, use it, else try to build from file id
            url = data_url
            if "drive.google.com" in data_url and "uc?" not in data_url:
                # attempt to extract ID
                import re
                m = re.search(r"/d/([a-zA-Z0-9_-]+)", data_url)
                if m:
                    file_id = m.group(1)
                    url = f"https://drive.google.com/uc?id={file_id}"
            # download if not exists
            if not os.path.exists(local_filename):
                gdown.download(url, local_filename, quiet=False)
            df = pd.read_csv(local_filename)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_url}: {e}") from e

    # sanitize dataframe
    df.columns = [c.strip() for c in df.columns]
    # ensure lat/lon columns present; try common names
    if "lat" not in df.columns or "lon" not in df.columns:
        # attempt to extract from .geo if present
        if ".geo" in df.columns:
            import json as _json
            def _extract(g):
                try:
                    obj = _json.loads(g)
                    lon, lat = obj.get("coordinates", (None, None))
                    return lat, lon
                except:
                    return None, None
            latlon = df[".geo"].apply(lambda g: pd.Series(_extract(g)))
            latlon.columns = ["lat", "lon"]
            df = pd.concat([df, latlon], axis=1)

    # convert date and numeric fields
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # numeric conversion
    for col in df.columns:
        if col not in ["date", ".geo"]:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass

    # drop rows missing essential fields
    df = df.dropna(subset=["date", "lat", "lon"], how="any")
    df = df.reset_index(drop=True)
    return df

# ----------------------------
# Helpers: load Kerala polygon (shapely only)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_kerala_polygon(geojson_path=BOUNDARY_PATH):
    # Accept local path or raw GitHub URL
    try:
        if os.path.exists(geojson_path):
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
        else:
            # try to fetch remote URL
            import urllib.request
            with urllib.request.urlopen(geojson_path) as resp:
                gj = json.load(resp)
    except Exception as e:
        raise RuntimeError(f"Failed to load Kerala boundary from {geojson_path}: {e}") from e

    # normalize features
    features = []
    if isinstance(gj, dict) and "features" in gj:
        features = gj["features"]
    elif isinstance(gj, list):
        features = gj
    else:
        features = [gj]

    polys = []
    from shapely.ops import unary_union
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat
        if geom is None:
            continue
        shp = shape(geom)
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)

    if not polys:
        raise RuntimeError("No polygon found in the provided GeoJSON.")

    if len(polys) == 1:
        return polys[0]
    else:
        return unary_union(polys)

# ----------------------------
# Clip points to polygon (vectorized)
# ----------------------------
def clip_points_to_polygon(df, polygon):
    pts = [Point(xy) for xy in zip(df["lon"].values, df["lat"].values)]
    mask = np.array([polygon.contains(p) for p in pts])
    return df.loc[mask].reset_index(drop=True)

# ----------------------------
# Kriging function
# ----------------------------
def do_ordinary_kriging(df_points, pollutant, grid_res=150, variogram_model="spherical"):
    lons = df_points["lon"].values.astype(float)
    lats = df_points["lat"].values.astype(float)
    vals = df_points[pollutant].values.astype(float)

    # simple check
    if len(vals) < 3:
        raise ValueError("Need at least 3 points to perform kriging")

    pad_lon = (lons.max() - lons.min()) * 0.02
    pad_lat = (lats.max() - lats.min()) * 0.02
    min_lon, max_lon = lons.min() - pad_lon, lons.max() + pad_lon
    min_lat, max_lat = lats.min() - pad_lat, lats.max() + pad_lat

    grid_lon = np.linspace(min_lon, max_lon, grid_res)
    grid_lat = np.linspace(min_lat, max_lat, grid_res)

    OK = OrdinaryKriging(
        lons, lats, vals,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    z, ss = OK.execute("grid", grid_lon, grid_lat)  # z: (ny, nx)
    return grid_lon, grid_lat, z

# ----------------------------
# Mask grid by polygon -> flat DataFrame
# ----------------------------
def mask_grid_by_polygon(grid_lon, grid_lat, z, polygon):
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
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

    grid_df = pd.DataFrame({"lon": inside_lon, "lat": inside_lat, "value": inside_z})
    return grid_df

# ----------------------------
# Plotting helper
# ----------------------------
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
        labels={"value": title}
    )
    fig.update_layout(mapbox_style="open-street-map", title=title)
    return fig

# ----------------------------
# MAIN app flow
# ----------------------------
st.title("Kerala Pollution Dashboard — Kriging Enabled")

with st.spinner("Loading data..."):
    try:
        df_all = load_data(DATA_URL, LOCAL_FILENAME)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    try:
        kerala_poly = load_kerala_polygon(BOUNDARY_PATH)
    except Exception as e:
        st.error(f"Failed to load Kerala boundary: {e}")
        st.stop()

# sidebar
st.sidebar.header("Controls")
# detect pollutant columns automatically if possible
possible_pollutants = [c for c in df_all.columns if c.upper() in ["AOD","NO2","SO2","CO","O3"]]
if not possible_pollutants:
    # fallback: any numeric column except lat/lon/date
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    possible_pollutants = [c for c in numeric_cols if c not in ["lat","lon"]]
pollutant = st.sidebar.selectbox("Select Pollutant", possible_pollutants, index=0)
mode = st.sidebar.radio("View Mode", ["Interactive Map", "Daily Animation", "Monthly Animation", "Heatmap", "Kriging Smooth Map"])
sample_size = st.sidebar.slider("Sample Size (for plotting/kriging)", 500, 5000, 2000, step=100)
grid_res = st.sidebar.slider("Kriging grid resolution (side length)", 80, 300, 150, step=10)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical", "exponential", "gaussian"], index=0)

# date filter
date_min = df_all["date"].min()
date_max = df_all["date"].max()
date_range = st.sidebar.date_input("Date range", [date_min.date(), date_max.date()], min_value=date_min.date(), max_value=date_max.date())

try:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_all[(df_all["date"] >= start_dt) & (df_all["date"] <= end_dt)]
except Exception:
    df_filtered = df_all.copy()

# clip to Kerala
df_clipped = clip_points_to_polygon(df_filtered, kerala_poly)
if df_clipped.empty:
    st.error("No points inside Kerala for selected date range.")
    st.stop()

st.sidebar.write(f"Points inside Kerala: {len(df_clipped):,}")

# sample for display/kriging
df_sample = df_clipped.sample(min(sample_size, len(df_clipped)), random_state=42)

st.title("🌏 Kerala Air Pollution Dashboard")
st.write(f"Showing pollutant: **{pollutant}** — Mode: **{mode}**")

# modes
if mode == "Interactive Map":
    st.subheader("Interactive pollution points")
    fig = px.scatter_mapbox(
        df_sample, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        hover_data=["date", pollutant],
        zoom=7, height=750, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Daily Animation":
    st.subheader("Daily Time-lapse")
    df_anim = df_sample.copy()
    df_anim["frame"] = df_anim["date"].dt.strftime("%Y-%m-%d")
    fig = px.scatter_mapbox(
        df_anim, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        animation_frame="frame",
        hover_data=["date", pollutant],
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Monthly Animation":
    st.subheader("Monthly Mean Time-lapse (sorted)")
    df_m = df_clipped.copy()
    df_m["year_month"] = df_m["date"].dt.to_period("M").astype(str)
    df_m["ym_dt"] = pd.to_datetime(df_m["year_month"], format="%Y-%m")
    df_month = df_m.groupby(["ym_dt", "lat", "lon"])[pollutant].mean().reset_index().sort_values("ym_dt")
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

elif mode == "Heatmap":
    st.subheader("Density heatmap")
    fig = px.density_mapbox(
        df_sample, lat="lat", lon="lon",
        z=pollutant, radius=25,
        center=dict(lat=df_clipped["lat"].mean(), lon=df_clipped["lon"].mean()),
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

else:  # Kriging Smooth Map
    st.subheader("Kriging — Ordinary Kriging (smooth map)")
    df_pts = df_clipped.dropna(subset=[pollutant, "lat", "lon"]).copy()
    if len(df_pts) > sample_size:
        df_pts = df_pts.sample(sample_size, random_state=42)

    st.write(f"Using {len(df_pts)} points. Grid: {grid_res}x{grid_res}. Variogram: {variogram_model}")

    if df_pts["lon"].nunique() < 3 or df_pts["lat"].nunique() < 3:
        st.error("Not enough distinct points for kriging.")
    else:
        with st.spinner("Performing kriging..."):
            try:
                grid_lon, grid_lat, z = do_ordinary_kriging(df_pts, pollutant, grid_res=grid_res, variogram_model=variogram_model)
            except Exception as e:
                st.error(f"Kriging failed: {e}")
                st.stop()

        grid_df = mask_grid_by_polygon(grid_lon, grid_lat, z, kerala_poly)
        if grid_df.empty:
            st.error("Kriged grid masked to polygon is empty. Try different parameters.")
        else:
            center_lat = df_clipped["lat"].mean()
            center_lon = df_clipped["lon"].mean()
            fig = plot_density_from_grid_df(grid_df, f"Kriged {pollutant}", center_lat, center_lon, zoom=7)

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
st.write("Notes: Kriging can be CPU-heavy. Tune sample size and grid resolution for performance.")
