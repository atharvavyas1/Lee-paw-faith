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

# Create map rendering function using fragment to prevent reloads
@st.fragment
def render_map(df_filtered, max_points, use_clustering):
    if len(df_filtered) == 0:
        st.warning("No records match the selected filters.")
        return
    
    # Limit data for performance
    df_map = df_filtered.head(max_points) if len(df_filtered) > max_points else df_filtered
    
    if len(df_filtered) > max_points:
        st.info(f"Showing {max_points} of {len(df_filtered)} points for performance. Use filters to refine data.")
    
    # Calculate center point
    center_lat = df_map['latitude'].mean()
    center_lon = df_map['longitude'].mean()
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap',
        prefer_canvas=True  # Better performance for many markers
    )
    
    # Define color mapping for all outcome types
    outcome_colors = {
        'adoption': 'green',
        'transfer': 'blue', 
        'return to owner': 'purple',
        'rtos': 'purple',
        'euthanize': 'red',
        'disposal': 'black',
        'died': 'darkred',
        'vet': 'orange',
        'escaped/stolen': 'pink',
        'shelter': 'lightblue'
    }
    
    if use_clustering:
        # Use MarkerCluster for better performance with many points
        from folium.plugins import MarkerCluster
        marker_cluster = MarkerCluster(
            name="Animal Locations",
            overlay=True,
            control=True,
            maxClusterRadius=50
        ).add_to(m)
        
        # Add markers to cluster
        for idx, row in df_map.iterrows():
            # Create popup text with available information
            popup_text = f"<b>Record ID:</b> {idx}<br>"
            
            # Add available columns to popup
            for col in ['animal_type', 'breed', 'color', 'outcome_type', 'age_upon_outcome']:
                if col in row and pd.notna(row[col]):
                    popup_text += f"<b>{col.replace('_', ' ').title()}:</b> {row[col]}<br>"
            
            popup_text += f"<b>Coordinates:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})"
            
            # Determine marker color
            color = 'gray'  # default
            if 'outcome_type' in row and pd.notna(row['outcome_type']):
                outcome = str(row['outcome_type']).lower().strip()
                color = outcome_colors.get(outcome, 'gray')
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='paw', prefix='fa')
            ).add_to(marker_cluster)
    
    else:
        # Use CircleMarkers for better performance (lighter weight)
        for idx, row in df_map.iterrows():
            # Create popup text
            popup_text = f"<b>Record ID:</b> {idx}<br>"
            for col in ['animal_type', 'breed', 'color', 'outcome_type', 'age_upon_outcome']:
                if col in row and pd.notna(row[col]):
                    popup_text += f"<b>{col.replace('_', ' ').title()}:</b> {row[col]}<br>"
            popup_text += f"<b>Coordinates:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})"
            
            # Determine color
            color = '#808080'  # default gray
            if 'outcome_type' in row and pd.notna(row['outcome_type']):
                outcome = str(row['outcome_type']).lower().strip()
                color_map = {
                    'adoption': '#28a745', 'transfer': '#007bff', 'return to owner': '#6f42c1',
                    'rtos': '#6f42c1', 'euthanize': '#dc3545', 'disposal': '#343a40',
                    'died': '#721c24', 'vet': '#fd7e14', 'escaped/stolen': '#e83e8c',
                    'shelter': '#17a2b8'
                }
                color = color_map.get(outcome, '#808080')
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='white',
                weight=1,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 300px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4>Outcome Types</h4>
    <p><span style="color: #28a745;">●</span><span style="color: #28a745;"> Adoption</span></p>
    <p><span style="color: #007bff;">●</span><span style="color: #007bff;"> Transfer</span></p>
    <p><span style="color: #6f42c1;">●</span><span style="color: #6f42c1;"> Return to Owner/RTOS</span></p>
    <p><span style="color: #dc3545;">●</span><span style="color: #dc3545;"> Euthanize</span></p>
    <p><span style="color: #343a40;">●</span><span style="color: #343a40;"> Disposal</span></p>
    <p><span style="color: #721c24;">●</span><span style="color: #721c24;"> Died</span></p>
    <p><span style="color: #fd7e14;">●</span><span style="color: #fd7e14;"> Vet</span></p>
    <p><span style="color: #e83e8c;">●</span><span style="color: #e83e8c;"> Escaped/Stolen</span></p>
    <p><span style="color: #17a2b8;">●</span><span style="color: #17a2b8;"> Shelter</span></p>
    <p><span style="color: #808080;">●</span><span style="color: #808080;"> Other</span></p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map with a stable key to prevent reloads
    map_data = st_folium(m, width=700, height=500, key="animal_shelter_map")
    return map_data

# Initialize session state for stable filtering
if 'filter_animal_type' not in st.session_state:
    st.session_state.filter_animal_type = 'All'
if 'filter_outcome' not in st.session_state:
    st.session_state.filter_outcome = 'All'
if 'map_max_points' not in st.session_state:
    st.session_state.map_max_points = 250
if 'use_clustering' not in st.session_state:
    st.session_state.use_clustering = True

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
            selected_animal_type = st.sidebar.selectbox(
                "Animal Type", 
                animal_types,
                index=animal_types.index(st.session_state.filter_animal_type) if st.session_state.filter_animal_type in animal_types else 0,
                key="animal_type_select"
            )
            st.session_state.filter_animal_type = selected_animal_type
            
            if selected_animal_type != 'All':
                df_filtered = df_with_coords[df_with_coords['animal_type'] == selected_animal_type]
            else:
                df_filtered = df_with_coords
        else:
            df_filtered = df_with_coords
        
        # Outcome type filter (if available)
        if 'outcome_type' in df_filtered.columns:
            outcome_types = ['All'] + sorted(df_filtered['outcome_type'].dropna().unique().tolist())
            selected_outcome = st.sidebar.selectbox(
                "Outcome Type", 
                outcome_types,
                index=outcome_types.index(st.session_state.filter_outcome) if st.session_state.filter_outcome in outcome_types else 0,
                key="outcome_type_select"
            )
            st.session_state.filter_outcome = selected_outcome
            
            if selected_outcome != 'All':
                df_filtered = df_filtered[df_filtered['outcome_type'] == selected_outcome]
        
        # Create the map section
        st.subheader("📍 Animal Shelter Locations Map")
        
        # Add map performance controls
        col1, col2 = st.columns([3, 1])
        with col2:
            max_points = st.selectbox(
                "Max points to display",
                [100, 250, 500, 1000, 2500, 5000, 10000, 30000],
                index=[100, 250, 500, 1000, 2500].index(st.session_state.map_max_points),
                help="Limit points for better performance",
                key="max_points_select"
            )
            st.session_state.map_max_points = max_points
            
            use_clustering = st.checkbox(
                "Use clustering", 
                value=st.session_state.use_clustering, 
                help="Group nearby points",
                key="clustering_checkbox"
            )
            st.session_state.use_clustering = use_clustering
        
        # Render the map using the fragment function
        with st.spinner("Rendering map..."):
            map_data = render_map(df_filtered, max_points, use_clustering)
        
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
    - Data table showing filtered results
    
    **Marker Colors:**
    - 🟢 Green: Adoption
    - 🔵 Blue: Transfer  
    - 🟣 Purple: Return to Owner/RTOS
    - 🔴 Red: Euthanize
    - ⚫ Black: Disposal
    - 🟤 Dark Red: Died
    - 🟠 Orange: Vet
    - 🩷 Pink: Escaped/Stolen
    - 🔵 Light Blue: Shelter
    - ⚫ Gray: Other outcomes
    
    **Data Source:** Sonoma County Open Data Portal
    """)