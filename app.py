import streamlit as st
import pandas as pd
import requests
import io
import re
import numpy as np

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
chart_data = df[['latitude', 'longitude']].dropna()

st.write("Let's make a geospatial map of the Sonoma County Animal Shelter data")
st.map(chart_data) 