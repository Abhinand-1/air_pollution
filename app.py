import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

st.set_page_config(page_title="Kerala Pollution Dashboard", layout="wide")

# ------------------------------------------------
# 1️⃣ LOAD MAIN CSV FILE (Google Drive)
# ------------------------------------------------
@st.cache_data
def load_data():
    # Direct download link (Google Drive)
    gdrive_url = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"

    df = pd.read_csv(gdrive_url)

    # Fix date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df


# ------------------------------------------------
# 2️⃣ LOAD KERALA BOUNDARY (GeoJSON)
# ------------------------------------------------
@st.cache_data
def load_kerala_boundary():
    boundary_url = "https://raw.githubusercontent.com/Abhinand-1/air_pollution/main/kerala_boundary.geojson"

    kerala = gpd.read_file(boundary_url)
    kerala = kerala.to_crs("EPSG:4326")
    return kerala


# ------------------------------------------------
# 3️⃣ CLIP POINTS TO KERALA ONLY
# ------------------------------------------------
@st.cache_data
def clip_to_kerala(df):
    kerala = load_kerala_boundary()

    # Convert your lat/lon to geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    # Spatial clip
    clipped = gpd.sjoin(gdf, kerala, how="inner", predicate="within")

    # Drop geometry for plotting
    clipped = clipped.drop(columns=["geometry", "index_right"], errors="ignore")

    return clipped


# ------------------------------------------------
# 4️⃣ LOAD + CLIP DATA
# ------------------------------------------------
df = load_data()
df = clip_to_kerala(df)

# ------------------------------------------------
# 5️⃣ SIDEBAR UI CONTROLS
# ------------------------------------------------
st.sidebar.header("Controls")
pollutant = st.sidebar.selectbox("Select Pollutant", ["AOD","NO2","SO2","CO","O3"])
mode = st.sidebar.radio("View Mode", ["Interactive Map", "Daily Animation", "Monthly Animation", "Heatmap"])
sample_size = st.sidebar.slider("Sample Size", 1000, 10000, 5000, 500)

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date Range", [date_min, date_max])

try:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
except:
    pass

st.title("🌏 Kerala Pollution Dashboard")
st.write(f"Displaying: **{pollutant}**")

df_s = df.sample(min(sample_size, len(df)), random_state=42)

# ------------------------------------------------
# 6️⃣ INTERACTIVE MAP
# ------------------------------------------------
if mode == "Interactive Map":
    st.subheader("📍 Interactive Pollution Map")
    fig = px.scatter_mapbox(
        df_s, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        hover_data=["date","AOD","NO2","SO2","CO","O3"],
        zoom=7, height=750, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# 7️⃣ DAILY ANIMATION
# ------------------------------------------------
elif mode == "Daily Animation":
    st.subheader("🎞 Daily Time-Lapse Animation")
    df_s["frame"] = df_s["date"].dt.strftime("%Y-%m-%d")

    fig = px.scatter_mapbox(
        df_s, lat="lat", lon="lon", color=pollutant,
        size=pollutant, animation_frame="frame",
        zoom=7, height=800, color_continuous_scale="Turbo",
        hover_data=["date","AOD","NO2","SO2","CO","O3"]
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# 8️⃣ MONTHLY ANIMATION
# ------------------------------------------------
elif mode == "Monthly Animation":
    st.subheader("📅 Monthly Mean Animation")
    df_m = df.copy()
    df_m["year_month"] = df_m["date"].dt.to_period("M").astype(str)
    df_month = df_m.groupby(["year_month","lat","lon"])[pollutant].mean().reset_index()
    df_month_s = df_month.sample(min(sample_size, len(df_month)), random_state=42)

    fig = px.scatter_mapbox(
        df_month_s, lat="lat", lon="lon",
        color=pollutant, size=pollutant,
        animation_frame="year_month",
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# 9️⃣ HEATMAP
# ------------------------------------------------
elif mode == "Heatmap":
    st.subheader("🔥 Pollution Density Heatmap")
    fig = px.density_mapbox(
        df_s, lat="lat", lon="lon", z=pollutant,
        radius=25,
        center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
        zoom=7, height=800, color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Built with Streamlit + Plotly | Kerala Air Pollution Dashboard")
