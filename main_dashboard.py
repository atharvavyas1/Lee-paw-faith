import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Austin Animal Shelter Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .insight-box {
        background-color: #000000;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🐾 Austin Animal Shelter Analytics Dashboard</h1>', unsafe_allow_html=True)

# Data loading and preprocessing functions
@st.cache_data
def load_and_process_data():
    """Load and process the Austin Animal Shelter data"""
    
    # Load data
    intake_url = "https://data.austintexas.gov/resource/wter-evkm.csv?$limit=500000"
    df_intakes = pd.read_csv(intake_url)
    
    outcome_url = "https://data.austintexas.gov/resource/9t4d-g238.csv?$limit=500000"
    df_outcomes = pd.read_csv(outcome_url)
    
    # Data preprocessing
    df_intakes['datetime'] = pd.to_datetime(df_intakes['datetime'])
    df_intakes['monthyear'] = pd.to_datetime(df_intakes['datetime2']).dt.strftime('%m-%Y')
    df_intakes = df_intakes.drop(columns=['datetime2'])
    
    df_outcomes['date_of_birth'] = pd.to_datetime(df_outcomes['date_of_birth'])
    df_outcomes['monthyear'] = pd.to_datetime(df_outcomes['monthyear'], format='%m-%Y')
    df_outcomes['datetime'] = pd.to_datetime(df_outcomes['datetime'], format='mixed', utc=True)
    
    # Remove null values
    df_intakes = df_intakes.dropna(subset=['sex_upon_intake'])
    df_outcomes = df_outcomes.dropna(subset=['outcome_type', 'sex_upon_outcome', 'age_upon_outcome'])
    
    # Sort and merge
    df_intakes_sorted = df_intakes.sort_values('datetime')
    df_outcomes_sorted = df_outcomes.sort_values('datetime')
    
    combined_df = pd.merge(df_intakes_sorted, df_outcomes_sorted, on='animal_id', 
                          suffixes=('_intake', '_outcome'), how='inner')
    
    # Age conversion functions
    def convert_age_to_days(age_str, name_str=None):
        if pd.isna(age_str):
            return None
        
        age_str = age_str.lower()
        
        if age_str.strip() == "0 years":
            if name_str and "grams" in str(name_str).lower():
                return 0
            else:
                return None
        
        total_days = 0
        patterns = [
            (r'(\d+)\s*year', 365),
            (r'(\d+)\s*month', 30),
            (r'(\d+)\s*week', 7),
            (r'(\d+)\s*day', 1)
        ]
        
        for pattern, multiplier in patterns:
            matches = re.findall(pattern, age_str)
            for match in matches:
                total_days += int(match) * multiplier
        
        return total_days if total_days > 0 else None
    
    def convert_age_to_years(age_str, name_str=None):
        days = convert_age_to_days(age_str, name_str)
        return days / 365 if days is not None else None
    
    # Apply age conversions
    combined_df['age_upon_intake_days'] = combined_df.apply(
        lambda row: convert_age_to_days(row['age_upon_intake'], row.get('name')), axis=1)
    combined_df['age_upon_intake_years'] = combined_df.apply(
        lambda row: convert_age_to_years(row['age_upon_intake'], row.get('name')), axis=1)
    
    # Feature engineering functions
    def engineer_breed_features(df, breed_column='breed'):
        df = df.copy()
        
        # Mixed breed indicators
        df['is_mixed'] = df[breed_column].str.contains('Mix|/', case=False, na=False)
        df['num_breeds'] = df[breed_column].str.count('/') + 1
        df['num_breeds'] = df['num_breeds'].where(df['is_mixed'], 1)
        
        # Size categories
        toy_breeds = ['Chihuahua', 'Yorkshire', 'Toy', 'Maltese', 'Pomeranian', 'Papillon', 'Miniature']
        small_breeds = ['Terrier', 'Beagle', 'Cocker', 'Dachshund', 'Corgi', 'Pug', 'Shih Tzu']
        large_breeds = ['Retriever', 'Shepherd', 'Mastiff', 'Great Dane', 'Rottweiler', 'Great Pyrenees', 'Newfoundland']
        
        def categorize_size(breed_str):
            if pd.isna(breed_str):
                return 'Unknown'
            breed_str = str(breed_str)
            if any(toy in breed_str for toy in toy_breeds):
                return 'Toy'
            elif any(large in breed_str for large in large_breeds):
                return 'Large'
            elif any(small in breed_str for small in small_breeds):
                return 'Small'
            else:
                return 'Medium'
        
        df['size_category'] = df[breed_column].apply(categorize_size)
        
        # Working/sport groups
        working_breeds = ['Shepherd', 'Cattle Dog', 'Border Collie', 'Australian Kelpie', 'Husky', 'Malamute']
        sporting_breeds = ['Retriever', 'Pointer', 'Spaniel', 'Setter', 'Vizsla']
        
        df['is_working'] = df[breed_column].str.contains('|'.join(working_breeds), case=False, na=False)
        df['is_sporting'] = df[breed_column].str.contains('|'.join(sporting_breeds), case=False, na=False)
        df['is_terrier'] = df[breed_column].str.contains('Terrier', case=False, na=False)
        
        # Primary breed extraction
        def extract_primary_breed(breed_str):
            if pd.isna(breed_str):
                return None
            breed_str = str(breed_str)
            if '/' in breed_str:
                return breed_str.split('/')[0].strip()
            elif 'Mix' in breed_str:
                return breed_str.replace(' Mix', '').strip()
            else:
                return breed_str.strip()
        
        df['primary_breed'] = df[breed_column].apply(extract_primary_breed)
        df['breed_complexity'] = df['num_breeds'] + df['is_mixed'].astype(int)
        
        return df
    
    def engineer_location_features(df, location_column='found_location'):
        df = df.copy()
        
        def extract_city(location_str):
            if pd.isna(location_str):
                return None
            location_str = str(location_str).strip()
            
            if 'outside jurisdiction' in location_str.lower():
                return 'Outside Jurisdiction'
            
            if ' in ' in location_str:
                parts = location_str.split(' in ')
                if len(parts) > 1:
                    city_part = parts[1].split('(')[0].strip()
                    return city_part
            
            city_match = re.search(r'^([A-Za-z\s]+)(?:\s*\(|$)', location_str)
            if city_match:
                return city_match.group(1).strip()
            
            return location_str
        
        df['city_area'] = df[location_column].apply(extract_city)
        
        # Austin area classifications
        austin_areas = ['Austin', 'Travis', 'Manor', 'Pflugerville', 'Cedar Park', 'Round Rock']
        df['is_austin_metro'] = df['city_area'].isin(austin_areas)
        df['is_core_austin'] = df['city_area'] == 'Austin'
        
        return df
    
    # Apply feature engineering
    combined_df = engineer_breed_features(combined_df, 'breed_outcome')
    combined_df = engineer_location_features(combined_df, 'found_location')
    
    # Clean up data
    combined_df['has_name'] = combined_df['name_outcome'].notnull().astype(int)
    combined_df = combined_df[combined_df['animal_type_outcome'].isin(['Dog', 'Cat'])]
    combined_df['Is_adopted'] = (combined_df['outcome_type'] == 'Adoption').astype(int)
    
    # Time features
    combined_df['intake_year'] = combined_df['datetime_intake'].dt.year
    combined_df['intake_month'] = combined_df['datetime_intake'].dt.month
    combined_df['intake_dayofweek'] = combined_df['datetime_intake'].dt.dayofweek
    combined_df['intake_is_weekend'] = (combined_df['intake_dayofweek'] >= 5).astype(int)
    
    # Visit count features
    combined_df = combined_df.sort_values(['animal_id', 'datetime_intake'])
    combined_df['visit_count'] = combined_df.groupby('animal_id').cumcount() + 1
    combined_df['is_return_visit'] = (combined_df['visit_count'] > 1).astype(int)
    
    # Clean final dataset
    combined_df = combined_df.dropna(subset=['age_upon_intake_days'])
    
    return combined_df

def run_isolation_forest(df):
    """Run isolation forest for outlier detection"""
    
    # Prepare categorical encodings
    categorical_columns = ['intake_type', 'intake_condition', 'animal_type_intake', 
                          'sex_upon_intake', 'size_category']
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Feature selection for outlier detection
    feature_columns = []
    
    # Add encoded categorical features
    for col in categorical_columns:
        if col + '_encoded' in df_encoded.columns:
            feature_columns.append(col + '_encoded')
    
    # Add numerical features
    numerical_features = ['age_upon_intake_days', 'num_breeds', 'breed_complexity', 
                         'has_name', 'visit_count', 'intake_month', 'intake_year']
    
    for feat in numerical_features:
        if feat in df_encoded.columns:
            feature_columns.append(feat)
    
    # Add boolean features
    boolean_features = ['is_mixed', 'is_working', 'is_sporting', 'is_terrier', 
                       'is_austin_metro', 'is_core_austin', 'is_return_visit']
    
    for feat in boolean_features:
        if feat in df_encoded.columns:
            df_encoded[feat] = df_encoded[feat].astype(int)
            feature_columns.append(feat)
    
    # Create feature matrix
    X = df_encoded[feature_columns].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100
    )
    
    outlier_labels = isolation_forest.fit_predict(X_scaled)
    outlier_scores = isolation_forest.score_samples(X_scaled)
    
    # Add results to dataframe
    df_encoded['outlier_label'] = outlier_labels
    df_encoded['outlier_score'] = outlier_scores
    df_encoded['is_outlier'] = (outlier_labels == -1)
    
    return df_encoded

# Load data
with st.spinner("Loading and processing data... This may take a moment."):
    df = load_and_process_data()

# Sidebar for filters
st.sidebar.header("🔍 Filters")

# Animal type filter
animal_types = ['All'] + list(df['animal_type_intake'].unique())
selected_animal_type = st.sidebar.selectbox("Animal Type", animal_types)

# Year filter
years = ['All'] + sorted(df['intake_year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Intake Year", years)

# Outcome filter
outcomes = ['All'] + list(df['outcome_type'].unique())
selected_outcome = st.sidebar.selectbox("Outcome Type", outcomes)

# Apply filters
filtered_df = df.copy()

if selected_animal_type != 'All':
    filtered_df = filtered_df[filtered_df['animal_type_intake'] == selected_animal_type]

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['intake_year'] == selected_year]

if selected_outcome != 'All':
    filtered_df = filtered_df[filtered_df['outcome_type'] == selected_outcome]

# Main dashboard content
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_animals = len(filtered_df)
    st.metric("Total Animals", f"{total_animals:,}")

with col2:
    adoption_rate = filtered_df['Is_adopted'].mean() * 100
    st.metric("Adoption Rate", f"{adoption_rate:.1f}%")

with col3:
    avg_age = filtered_df['age_upon_intake_days'].mean() / 365
    st.metric("Avg Age at Intake", f"{avg_age:.1f} years")

with col4:
    return_rate = filtered_df['is_return_visit'].mean() * 100
    st.metric("Return Visit Rate", f"{return_rate:.1f}%")

# EDA Insights Section
st.markdown("## 📊 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Adoption Insights</h4>
    <ul>
    <li>Younger animals have higher adoption rates</li>
    <li>Dogs have slightly higher adoption rates than cats</li>
    <li>Animals with names are more likely to be adopted</li>
    <li>Mixed breeds show good adoption outcomes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-box">
    <h4>📈 Intake Patterns</h4>
    <ul>
    <li>Summer months see higher intake volumes</li>
    <li>Weekends have fewer intakes than weekdays</li>
    <li>Austin metro area accounts for majority of intakes</li>
    <li>Stray animals make up the largest intake category</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Charts Section
st.markdown("## 📈 Interactive Visualizations")

# Row 1: Animal Type Distribution and Intake Over Time
col1, col2 = st.columns(2)

with col1:
    st.subheader("Animal Type Distribution")
    animal_counts = filtered_df['animal_type_intake'].value_counts()
    
    fig_pie = px.pie(
        values=animal_counts.values,
        names=animal_counts.index,
        title="Animal Intakes by Type",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Intakes Over Time")
    monthly_intakes = filtered_df.groupby(['intake_year', 'intake_month']).size().reset_index(name='count')
    monthly_intakes['date'] = pd.to_datetime(monthly_intakes.rename(columns={'intake_year': 'year', 'intake_month': 'month'}).assign(day=1)[['year', 'month', 'day']])
    
    fig_line = px.line(
        monthly_intakes, 
        x='date', 
        y='count',
        title="Monthly Intake Trends",
        markers=True
    )
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Number of Intakes")
    st.plotly_chart(fig_line, use_container_width=True)

# Row 2: Intake Conditions and Outcome Types
col1, col2 = st.columns(2)

with col1:
    st.subheader("Intake Conditions")
    condition_counts = filtered_df['intake_condition'].value_counts()
    
    fig_bar = px.bar(
        x=condition_counts.index,
        y=condition_counts.values,
        title="Animals by Intake Condition",
        color=condition_counts.values,
        color_continuous_scale="viridis"
    )
    fig_bar.update_layout(xaxis_title="Condition", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Outcome Types")
    outcome_counts = filtered_df['outcome_type'].value_counts()
    
    fig_bar2 = px.bar(
        x=outcome_counts.index,
        y=outcome_counts.values,
        title="Animals by Outcome Type",
        color=outcome_counts.values,
        color_continuous_scale="plasma"
    )
    fig_bar2.update_layout(xaxis_title="Outcome", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_bar2, use_container_width=True)

# Row 3: Age Distribution and Breed Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution at Intake")
    age_bins = pd.cut(filtered_df['age_upon_intake_days']/365, 
                     bins=[0, 1, 3, 7, 15, 25], 
                     labels=['Puppy/Kitten', 'Young', 'Adult', 'Senior', 'Geriatric'])
    age_counts = age_bins.value_counts()
    
    fig_age = px.bar(
        x=age_counts.index,
        y=age_counts.values,
        title="Age Groups at Intake",
        color=age_counts.values,
        color_continuous_scale="blues"
    )
    fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Top Breeds")
    if 'primary_breed' in filtered_df.columns:
        breed_counts = filtered_df['primary_breed'].value_counts().head(10)
        
        fig_breed = px.bar(
            x=breed_counts.values,
            y=breed_counts.index,
            orientation='h',
            title="Top 10 Primary Breeds",
            color=breed_counts.values,
            color_continuous_scale="greens"
        )
        fig_breed.update_layout(xaxis_title="Count", yaxis_title="Breed", showlegend=False)
        st.plotly_chart(fig_breed, use_container_width=True)

# Isolation Forest Section
st.markdown("## 🔍 Outlier Detection Analysis")

if st.button("Run Outlier Detection"):
    with st.spinner("Running Isolation Forest analysis..."):
        df_with_outliers = run_isolation_forest(filtered_df)
    
    col1, col2, col3 = st.columns(3)
    
    n_outliers = df_with_outliers['is_outlier'].sum()
    outlier_percentage = (n_outliers / len(df_with_outliers)) * 100
    avg_outlier_score = df_with_outliers['outlier_score'].mean()
    
    with col1:
        st.metric("Outliers Detected", f"{n_outliers:,}")
    
    with col2:
        st.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
    
    with col3:
        st.metric("Avg Outlier Score", f"{avg_outlier_score:.3f}")
    
    # Outlier Score Distribution
    st.subheader("Outlier Score Distribution")
    
    fig_outlier = px.histogram(
        df_with_outliers,
        x='outlier_score',
        nbins=50,
        title="Distribution of Outlier Scores",
        color_discrete_sequence=["skyblue"]
    )
    
    # Add threshold line
    threshold = df_with_outliers[df_with_outliers['is_outlier']]['outlier_score'].max()
    fig_outlier.add_vline(x=threshold, line_dash="dash", line_color="red", 
                         annotation_text="Outlier Threshold")
    
    st.plotly_chart(fig_outlier, use_container_width=True)
    
    # Outlier Analysis by Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Outliers by Animal Type")
        outlier_by_type = df_with_outliers[df_with_outliers['is_outlier']]['animal_type_intake'].value_counts()
        
        fig_outlier_type = px.pie(
            values=outlier_by_type.values,
            names=outlier_by_type.index,
            title="Outlier Distribution by Animal Type"
        )
        st.plotly_chart(fig_outlier_type, use_container_width=True)
    
    with col2:
        st.subheader("Outliers by Intake Condition")
        outlier_by_condition = df_with_outliers[df_with_outliers['is_outlier']]['intake_condition'].value_counts()
        
        fig_outlier_condition = px.bar(
            x=outlier_by_condition.index,
            y=outlier_by_condition.values,
            title="Outliers by Intake Condition",
            color=outlier_by_condition.values,
            color_continuous_scale="reds"
        )
        st.plotly_chart(fig_outlier_condition, use_container_width=True)
    
    # Outlier vs Normal Comparison
    st.subheader("Outlier Characteristics")
    
    comparison_features = ['age_upon_intake_days', 'visit_count', 'has_name', 'is_mixed']
    
    outlier_means = df_with_outliers[df_with_outliers['is_outlier']][comparison_features].mean()
    normal_means = df_with_outliers[~df_with_outliers['is_outlier']][comparison_features].mean()
    
    comparison_df = pd.DataFrame({
        'Feature': comparison_features,
        'Outliers': outlier_means.values,
        'Normal': normal_means.values
    })
    
    fig_comparison = px.bar(
        comparison_df.melt(id_vars='Feature', var_name='Group', value_name='Value'),
        x='Feature',
        y='Value',
        color='Group',
        barmode='group',
        title="Feature Comparison: Outliers vs Normal Cases"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Top outliers table
    st.subheader("Most Extreme Outliers")
    outlier_display_cols = ['animal_type_intake', 'age_upon_intake_days', 'intake_condition', 
                           'outcome_type', 'outlier_score']
    
    if all(col in df_with_outliers.columns for col in outlier_display_cols):
        top_outliers = df_with_outliers[df_with_outliers['is_outlier']].nsmallest(10, 'outlier_score')
        st.dataframe(top_outliers[outlier_display_cols])

# Footer
st.markdown("---")
st.markdown("**Data Source:** Austin Animal Center | **Dashboard Created with:** Streamlit & Plotly")
st.markdown("*This dashboard provides insights into Austin Animal Shelter operations and helps identify unusual cases through outlier detection.*")