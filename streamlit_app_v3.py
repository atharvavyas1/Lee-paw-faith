import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re
import pydeck as pdk

#st.write("🔄 App loaded at:", pd.Timestamp.now())

# Set page config
st.set_page_config(
    page_title="Local Urban Companion Archive - LUCA",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Local Urban Companion Archive (LUCA)")
st.markdown("Interactive map showing outcomes of animals who were admitted to the Sonoma County Animal Shelter")

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

def get_color_for_outcome(outcome):
    """Return RGB color array for outcome type"""
    color_map = {
        'adoption': [40, 167, 69],      # Green
        'transfer': [0, 123, 255],      # Blue
        'return to owner': [111, 66, 193],  # Purple
        'rtos': [111, 66, 193],         # Purple
        'euthanize': [220, 53, 69],     # Red
        'disposal': [52, 58, 64],       # Black
        'died': [114, 28, 36],          # Dark Red
        'vet': [253, 126, 20],          # Orange
        'escaped/stolen': [232, 62, 140], # Pink
        'shelter': [23, 162, 184]       # Light Blue
    }
    
    if pd.isna(outcome):
        return [128, 128, 128]  # Gray for NaN
    
    outcome_clean = str(outcome).lower().strip()
    return color_map.get(outcome_clean, [128, 128, 128])  # Default gray

# Create map rendering function using fragment to prevent reloads
@st.fragment
def render_pydeck_map(df_filtered, max_points, map_style):
    if len(df_filtered) == 0:
        st.warning("No records match the selected filters.")
        return
    
    # Limit data for performance
    df_map = df_filtered.head(max_points) if len(df_filtered) > max_points else df_filtered
    
    if len(df_filtered) > max_points:
        st.info(f"Showing {max_points} of {len(df_filtered)} points for performance. Use filters to refine data.")
    
    # Prepare data for pydeck
    df_map = df_map.copy()
    
    # Add color column based on outcome type
    df_map['color'] = df_map['outcome_type'].apply(get_color_for_outcome)
    
    # Create tooltip text
    df_map['tooltip'] = df_map.apply(lambda row: 
        f"Animal Type: {row.get('type', 'N/A')}\n" +
        f"Breed: {row.get('breed', 'N/A')}\n" +
        f"Color: {row.get('color', 'N/A')}\n" +
        f"Outcome: {row.get('outcome_type', 'N/A')}\n" +
        f"Time in Shelter: {row.get('time_in_shelter_days', 'N/A')} days\n" +
        f"Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})", 
        axis=1
    )
    
    # Calculate center point
    center_lat = df_map['latitude'].mean()
    center_lon = df_map['longitude'].mean()
    
    # Create the scatter plot layer
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_map,
        get_position=['longitude', 'latitude'],
        get_color='color',
        get_radius=100,  # Radius in meters
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        get_line_color=[255, 255, 255]  # White outline
    )
    
    # Define the view state
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=0,
        bearing=0
    )
    
    # Create the deck
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={
            'html': '<b>{tooltip}</b>',
            'style': {
                'backgroundColor': 'steelblue',
                'color': 'white',
                'fontSize': '12px',
                'padding': '10px',
                'borderRadius': '5px'
            }
        }
    )
    
    # Display the map
    st.pydeck_chart(deck, use_container_width=True)

# Initialize session state for stable filtering
if 'filter_animal_type' not in st.session_state:
    st.session_state.filter_animal_type = 'All'
if 'filter_outcome' not in st.session_state:
    st.session_state.filter_outcome = 'All'
if 'map_max_points' not in st.session_state:
    st.session_state.map_max_points = 1000
if 'map_style' not in st.session_state:
    st.session_state.map_style = 'mapbox://styles/mapbox/dark-v9'
if 'date_range' not in st.session_state:
    st.session_state.date_range = None

# Load and process data
with st.spinner("Loading Sonoma county animal shelter data..."):
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
    
    # Convert outcome_date column and drop rows with invalid dates
    if 'outcome_date' in df_with_coords.columns:
        df_with_coords = df_with_coords.copy()
        df_with_coords['outcome_date'] = pd.to_datetime(df_with_coords['outcome_date'], errors='coerce')
        df_with_coords.dropna(subset=['outcome_date'], inplace=True)
    
    # Convert intake_date column and drop rows with invalid dates
    if 'intake_date' in df_with_coords.columns:
        df_with_coords['intake_date'] = pd.to_datetime(df_with_coords['intake_date'], errors='coerce')
        # Don't drop rows with missing intake_date, just calculate where possible
    
    # Calculate time in shelter (in days)
    if 'outcome_date' in df_with_coords.columns and 'intake_date' in df_with_coords.columns:
        df_with_coords['time_in_shelter_days'] = (
            df_with_coords['outcome_date'] - df_with_coords['intake_date']
        ).dt.days
        # Replace negative values with NaN (data errors)
        df_with_coords.loc[df_with_coords['time_in_shelter_days'] < 0, 'time_in_shelter_days'] = np.nan

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
        
        # Start with the full dataset for filtering
        df_filtered = df_with_coords

        # Date range slider
        if 'outcome_date' in df_filtered.columns:
            min_date = df_with_coords['outcome_date'].min().date()
            max_date = df_with_coords['outcome_date'].max().date()

            if st.session_state.date_range is None:
                st.session_state.date_range = (min_date, max_date)
            
            # Ensure session state value is within the new data's bounds
            session_start, session_end = st.session_state.date_range
            if session_start < min_date or session_end > max_date:
                st.session_state.date_range = (min_date, max_date)

            selected_date_range = st.sidebar.slider(
                "Filter by Outcome Date",
                min_value=min_date,
                max_value=max_date,
                value=st.session_state.date_range,
                key="date_range_selector"
            )
            st.session_state.date_range = selected_date_range
            
            if len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                df_filtered = df_filtered[
                    (df_filtered['outcome_date'].dt.date >= start_date) & 
                    (df_filtered['outcome_date'].dt.date <= end_date)
                ]

        # Animal type filter (if available)
        if 'type' in df_with_coords.columns:
            animal_types = ['All'] + sorted(df_with_coords['type'].dropna().unique().tolist())
            selected_animal_type = st.sidebar.selectbox(
                "Animal Type", 
                animal_types,
                index=animal_types.index(st.session_state.filter_animal_type) if st.session_state.filter_animal_type in animal_types else 0,
                key="animal_type_select"
            )
            st.session_state.filter_animal_type = selected_animal_type
            
            if selected_animal_type != 'All':
                df_filtered = df_filtered[df_filtered['type'] == selected_animal_type]
        
        # Outcome type filter (if available)
        if 'outcome_type' in df_filtered.columns:
            outcome_types = ['All'] + sorted(df_with_coords['outcome_type'].dropna().unique().tolist())
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
        st.subheader("Outcome map for animals at the Sonoma County Animal Shelter")
        
        # Add map performance controls
        col1, col2 = st.columns([2, 1])
        with col2:
            max_points = st.selectbox(
                "Max points to display",
                [100, 250, 500, 1000, 2500, 5000, 10000, 30000],
                index=7,
                help="Limit points for better performance",
                key="max_points_select"
            )
            st.session_state.map_max_points = max_points
            
            map_styles = {
                'Dark': 'mapbox://styles/mapbox/dark-v9',
                'Light': 'mapbox://styles/mapbox/light-v9',
                'Satellite': 'mapbox://styles/mapbox/satellite-v9',
                'Streets': 'mapbox://styles/mapbox/streets-v11'
            }
            
            selected_style_name = st.selectbox(
                "Map Style",
                list(map_styles.keys()),
                index=list(map_styles.values()).index(st.session_state.map_style) if st.session_state.map_style in map_styles.values() else 0,
                key="map_style_select"
            )
            st.session_state.map_style = map_styles[selected_style_name]
        
        # Create color legend
        with col1:
            st.markdown("**Color Legend:**")
            legend_cols = st.columns(5)
            
            legend_items = [
                ("🟢 Adoption", "green"), ("🔵 Transfer", "blue"), 
                ("🟣 Return/RTOS", "purple"), ("🔴 Euthanize", "red"),
                ("⚫ Disposal", "black")
            ]
            
            for i, (label, color) in enumerate(legend_items):
                with legend_cols[i % 5]:
                    st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            
            legend_cols2 = st.columns(5)
            legend_items2 = [
                ("🟤 Died", "darkred"), ("🟠 Vet", "orange"),
                ("🩷 Escaped", "pink"), ("🔵 Shelter", "lightblue"),
                ("⚫ Other", "gray")
            ]
            
            for i, (label, color) in enumerate(legend_items2):
                with legend_cols2[i % 5]:
                    st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
        
        # Render the map using the fragment function
        with st.spinner("Rendering Pydeck map..."):
            render_pydeck_map(df_filtered, max_points, st.session_state.map_style)
        
        # Display filtered data table
        st.subheader("📊 Filtered Data")
        
        # Select relevant columns for display
        display_cols = ['latitude', 'longitude']
        for col in ['type', 'breed', 'color', 'outcome_type', 'time_in_shelter_days', 'outcome_date']:
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
with st.expander("ℹ️ About this Pydeck Dashboard"):
    st.markdown("""
    This dashboard displays Sonoma county animal shelter location data using **Pydeck** for high-performance visualization.
    
    **Features:**
    - Interactive map with colored points based on outcome type
    - Filters for animal type and outcome type
    - Multiple map styles (Light, Dark, Satellite, Streets)
    - Hover tooltips with detailed information
    - High performance with large datasets
    
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

    # Add a date slider on here
    # Rename the mapping tool to LUCA