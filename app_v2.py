import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(
    page_title="Animal Shelter Locations",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Animal Shelter Location Dashboard")
st.markdown("Interactive map showing animal shelter locations in Sonoma County")

# Function to extract the (lat, lon) string or keep NaN
def extract_latlon(val):
    if pd.isnull(val):
        return np.nan
    match = re.search(r'\(([^)]+)\)', val)
    if match:
        return f"({match.group(1)})"
    else:
        return np.nan

# Split into latitude and longitude columns
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

# Cache the data loading function for better performance
@st.cache_data
def load_data():
    # Download all data in batches (CSV API limit workaround)
    BASE_URL = "https://data.sonomacounty.ca.gov/resource/924a-vesw.csv"
    BATCH_SIZE = 1000
    offset = 0
    dfs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
        status_text.text(f"Fetching records... {offset} loaded")
        
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Concatenate all batches
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        st.success(f"Successfully loaded {len(df)} records!")
        return df
    else:
        st.error("No data fetched.")
        st.stop()

# Load and process data
with st.spinner("Loading animal shelter data..."):
    df = load_data()

# Clean location data
if 'location' in df.columns:
    # Apply extraction
    df['location_clean'] = df['location'].apply(extract_latlon)
    
    # Split into latitude and longitude columns
    df[['latitude', 'longitude']] = df['location_clean'].apply(split_latlon)
    df = df.drop(columns=['location_clean'])
    
    # Filter out rows with missing coordinates
    df_with_coords = df.dropna(subset=['latitude', 'longitude'])
    
    st.write(f"**Data Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Records with Coordinates", len(df_with_coords))
    with col3:
        st.metric("Missing Coordinates", len(df) - len(df_with_coords))
    
    if len(df_with_coords) > 0:
        # Create sidebar filters
        st.sidebar.header("Filters")
        
        # Animal type filter (if available)
        if 'animal_type' in df_with_coords.columns:
            animal_types = ['All'] + sorted(df_with_coords['animal_type'].dropna().unique().tolist())
            selected_animal_type = st.sidebar.selectbox("Animal Type", animal_types)
            
            if selected_animal_type != 'All':
                df_filtered = df_with_coords[df_with_coords['animal_type'] == selected_animal_type]
            else:
                df_filtered = df_with_coords
        else:
            df_filtered = df_with_coords
        
        # Outcome type filter (if available)
        if 'outcome_type' in df_filtered.columns:
            outcome_types = ['All'] + sorted(df_filtered['outcome_type'].dropna().unique().tolist())
            selected_outcome = st.sidebar.selectbox("Outcome Type", outcome_types)
            
            if selected_outcome != 'All':
                df_filtered = df_filtered[df_filtered['outcome_type'] == selected_outcome]
        
        # Create the map
        st.subheader("📍 Animal Shelter Locations Map")
        
        if len(df_filtered) > 0:
            # Calculate center point
            center_lat = df_filtered['latitude'].mean()
            center_lon = df_filtered['longitude'].mean()
            
            # Create folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add markers for each location
            for idx, row in df_filtered.iterrows():
                # Create popup text with available information
                popup_text = f"<b>Record ID:</b> {idx}<br>"
                
                # Add available columns to popup
                for col in ['animal_type', 'breed', 'color', 'outcome_type', 'age_upon_outcome']:
                    if col in row and pd.notna(row[col]):
                        popup_text += f"<b>{col.replace('_', ' ').title()}:</b> {row[col]}<br>"
                
                popup_text += f"<b>Coordinates:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})"
                
                # Determine marker color based on outcome type (if available)
                if 'outcome_type' in row and pd.notna(row['outcome_type']):
                    outcome = str(row['outcome_type']).lower()
                    if 'adoption' in outcome:
                        color = 'green'
                    elif 'transfer' in outcome:
                        color = 'blue'
                    elif 'return' in outcome:
                        color = 'orange'
                    elif 'euthanasia' in outcome:
                        color = 'red'
                    else:
                        color = 'gray'
                else:
                    color = 'blue'
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=color, icon='heart')
                ).add_to(m)
            
            # Display the map
            map_data = st_folium(m, width=700, height=500)
            
            # Display filtered data table
            st.subheader("📊 Filtered Data")
            
            # Select relevant columns for display
            display_cols = ['latitude', 'longitude']
            for col in ['animal_type', 'breed', 'color', 'outcome_type', 'age_upon_outcome', 'datetime']:
                if col in df_filtered.columns:
                    display_cols.append(col)
            
            st.dataframe(
                df_filtered[display_cols].head(100),  # Limit to first 100 rows for performance
                use_container_width=True
            )
            
            if len(df_filtered) > 100:
                st.info(f"Showing first 100 rows of {len(df_filtered)} filtered records")
        
        else:
            st.warning("No records match the selected filters.")
    
    else:
        st.error("No records with valid coordinates found in the dataset.")

else:
    st.error("Location column not found in the dataset. Please check the data source.")

# Add information about the dashboard
with st.expander("ℹ️ About this Dashboard"):
    st.markdown("""
    This dashboard displays animal shelter location data from Sonoma County.
    
    **Features:**
    - Interactive map with colored markers based on outcome type
    - Filters for animal type and outcome type
    - Detailed popup information for each location
    - Data table showing filtered results
    
    **Marker Colors:**
    - 🟢 Green: Adoptions
    - 🔵 Blue: Transfers
    - 🟠 Orange: Returns
    - 🔴 Red: Euthanasia
    - ⚫ Gray: Other outcomes
    
    **Data Source:** Sonoma County Open Data Portal
    """)