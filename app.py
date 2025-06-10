import streamlit as st
import pandas as pd
import requests
import io
import re
import numpy as np
import pydeck as pdk

st.title("Sonoma County Animal Shelter Geospatial Map")

# Download all data in batches (CSV API limit workaround)
BASE_URL = "https://data.sonomacounty.ca.gov/resource/924a-vesw.csv"
BATCH_SIZE = 1000
offset = 0
dfs = []

while True:
    params = {
        "$limit": BATCH_SIZE,
        "$offset": offset
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    if not response.text.strip():
        break
    batch_df = pd.read_csv(io.StringIO(response.text))
    if batch_df.empty:
        break
    dfs.append(batch_df)
    offset += BATCH_SIZE
    st.write(f"Fetched {offset} records...")

# Concatenate all batches
if dfs:
    df = pd.concat(dfs, ignore_index=True)
else:
    st.error("No data fetched.")
    st.stop()

# Clean the location column and extract latitude/longitude
def extract_latlon(val):
    if pd.isnull(val):
        return np.nan
    match = re.search(r'\(([^)]+)\)', str(val))
    if match:
        return f"({match.group(1)})"
    else:
        return np.nan

df['location_clean'] = df['location'].apply(extract_latlon)

def split_latlon(val):
    if pd.isnull(val):
        return pd.Series([np.nan, np.nan])
    latlon = val.strip('()').split(',')
    if len(latlon) == 2:
        try:
            return pd.Series([float(latlon[0]), float(latlon[1])])
        except ValueError:
            return pd.Series([np.nan, np.nan])
    else:
        return pd.Series([np.nan, np.nan])

df[['latitude', 'longitude']] = df['location_clean'].apply(split_latlon)

# Prepare data for map (drop rows with missing lat/lon)
chart_data = df[['latitude', 'longitude', 'outcome_type']].dropna()

st.write("Let's make a geospatial map of the Sonoma County Animal Shelter data")
# st.map(chart_data) 

# Assign a unique color to each outcome_type
outcome_types = chart_data['outcome_type'].unique()
color_palette = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 0, 128],  # Purple
    [255, 165, 0],  # Orange
    [0, 128, 0],    # Dark Green
    [128, 128, 128] # Gray
]
color_map = {ot: color_palette[i % len(color_palette)] for i, ot in enumerate(outcome_types)}
chart_data['color'] = chart_data['outcome_type'].map(color_map)

# Convert DataFrame to records for pydeck
data_for_pydeck = chart_data.to_dict(orient='records')

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data_for_pydeck,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=100,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=chart_data['latitude'].mean(),
    longitude=chart_data['longitude'].mean(),
    zoom=9,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))