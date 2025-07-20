import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Austin Animal Shelter Predictions",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🐕 Austin Animal Shelter Prediction Dashboard")
st.markdown("Predict future animal intakes and outcomes using machine learning models")

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")
prediction_type = st.sidebar.selectbox(
    "Select Prediction Type:",
    ["Animal Intakes", "Animal Outcomes"]
)

months_ahead = st.sidebar.slider(
    "Months to Predict:",
    min_value=1,
    max_value=6,
    value=3,
    help="Select how many months into the future to predict"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the data"""
    # Load data
    intake_url = "https://data.austintexas.gov/resource/wter-evkm.csv?$limit=500000"
    outcome_url = "https://data.austintexas.gov/resource/9t4d-g238.csv?$limit=500000"
    
    df_intakes = pd.read_csv(intake_url)
    df_outcomes = pd.read_csv(outcome_url)
    
    # Convert datetime columns
    df_intakes['datetime'] = pd.to_datetime(df_intakes['datetime'])
    df_outcomes['datetime'] = pd.to_datetime(df_outcomes['datetime'])
    
    # Extract date
    df_intakes['date'] = df_intakes['datetime'].dt.to_period('M')
    df_outcomes['date'] = df_outcomes['datetime'].dt.to_period('M')
    
    # Create aggregated data
    aggregated_intakes = df_intakes.groupby(['date', 'animal_type', 'intake_type']).size().reset_index(name='count')
    aggregated_outcomes = df_outcomes.groupby(['date', 'animal_type', 'outcome_type']).size().reset_index(name='count')
    
    # Convert period back to datetime for processing
    aggregated_intakes['date'] = aggregated_intakes['date'].dt.to_timestamp()
    aggregated_outcomes['date'] = aggregated_outcomes['date'].dt.to_timestamp()
    
    return df_intakes, df_outcomes, aggregated_intakes, aggregated_outcomes

# Feature engineering functions
def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12]):
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[3, 6, 12]):
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    return df

def create_seasonal_features(df, date_col):
    df = df.copy()
    df['month'] = pd.to_datetime(df[date_col]).dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    return df

def prepare_intake_features(df, intake_col='count', date_col='date'):
    df = df.copy()
    df = create_seasonal_features(df, date_col)
    df = create_lag_features(df, intake_col, lags=[1, 2, 3, 6, 12])
    df = create_rolling_features(df, intake_col, windows=[3, 6, 12])
    df[f'{intake_col}_yoy'] = df[intake_col] / df[intake_col].shift(12)
    return df

# Model classes
class XGBoostIntakeForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
        )
        self.feature_cols = None
        
    def prepare_data(self, df, target_col='count', test_size=0.2):
        df_sorted = df.sort_values('date').reset_index(drop=True)
        exclude_cols = [target_col, 'date', 'year']
        self.feature_cols = [col for col in df_sorted.columns if col not in exclude_cols]
        
        X = df_sorted[self.feature_cols]
        y = df_sorted[target_col]
        
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, df_sorted
    
    def fit_and_evaluate(self, df, target_col='count'):
        X_train, X_test, y_train, y_test, df_sorted = self.prepare_data(df, target_col)
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        return df_sorted, X_train, X_test, y_train, y_test, test_pred
    
    def predict_future(self, df, target_col='count', months_ahead=3):
        # Fit the model on all available data
        df_sorted = df.sort_values('date').reset_index(drop=True)
        exclude_cols = [target_col, 'date', 'year']
        self.feature_cols = [col for col in df_sorted.columns if col not in exclude_cols]
        
        X = df_sorted[self.feature_cols].fillna(method='ffill').fillna(0)
        y = df_sorted[target_col]
        
        self.model.fit(X, y)
        
        # Create future dates
        last_date = df_sorted['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=months_ahead, freq='MS')
        
        # Prepare future data
        future_data = []
        current_df = df_sorted.copy()
        
        for future_date in future_dates:
            # Create a new row for prediction
            new_row = {'date': future_date, target_col: np.nan}
            
            # Add to current data
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Recreate features with the new data point
            current_df = prepare_intake_features(current_df, target_col, 'date')
            
            # Get the features for the last row (future prediction)
            feature_cols = [col for col in current_df.columns if col not in [target_col, 'date', 'year']]
            future_features = current_df[feature_cols].iloc[-1:].fillna(method='ffill').fillna(0)
            
            # Make prediction
            prediction = self.model.predict(future_features)[0]
            
            # Update the target value with prediction for next iteration
            current_df.iloc[-1, current_df.columns.get_loc(target_col)] = prediction
            future_data.append(prediction)
        
        return future_dates, future_data

class XGBoostOutcomeForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            reg_alpha=2.0,
            reg_lambda=2.0,
            random_state=42
        )
        self.feature_cols = None
        
    def prepare_data(self, df, target_col='count', test_size=0.2):
        df_sorted = df.sort_values('date').reset_index(drop=True)
        exclude_cols = [target_col, 'date', 'year']
        self.feature_cols = [col for col in df_sorted.columns if col not in exclude_cols]
        
        X = df_sorted[self.feature_cols]
        y = df_sorted[target_col]
        
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, df_sorted
    
    def fit_and_evaluate(self, df, target_col='count'):
        X_train, X_test, y_train, y_test, df_sorted = self.prepare_data(df, target_col)
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        return df_sorted, X_train, X_test, y_train, y_test, test_pred
    
    def predict_future(self, df, target_col='count', months_ahead=3):
        # Fit the model on all available data
        df_sorted = df.sort_values('date').reset_index(drop=True)
        exclude_cols = [target_col, 'date', 'year']
        self.feature_cols = [col for col in df_sorted.columns if col not in exclude_cols]
        
        X = df_sorted[self.feature_cols].fillna(method='ffill').fillna(0)
        y = df_sorted[target_col]
        
        self.model.fit(X, y)
        
        # Create future dates
        last_date = df_sorted['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=months_ahead, freq='MS')
        
        # Prepare future data
        future_data = []
        current_df = df_sorted.copy()
        
        for future_date in future_dates:
            # Create a new row for prediction
            new_row = {'date': future_date, target_col: np.nan}
            
            # Add to current data
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Recreate features with the new data point
            current_df = prepare_intake_features(current_df, target_col, 'date')
            
            # Get the features for the last row (future prediction)
            feature_cols = [col for col in current_df.columns if col not in [target_col, 'date', 'year']]
            future_features = current_df[feature_cols].iloc[-1:].fillna(method='ffill').fillna(0)
            
            # Make prediction
            prediction = self.model.predict(future_features)[0]
            
            # Update the target value with prediction for next iteration
            current_df.iloc[-1, current_df.columns.get_loc(target_col)] = prediction
            future_data.append(prediction)
        
        return future_dates, future_data

# Main app logic
def main():
    # Load data
    with st.spinner("Loading data..."):
        df_intakes, df_outcomes, aggregated_intakes, aggregated_outcomes = load_data()
    
    # Filter data
    aggregated_intakes_filtered = aggregated_intakes[
        (aggregated_intakes['animal_type'].isin(['Dog', 'Cat'])) & 
        (aggregated_intakes['intake_type'].isin(['Stray', 'Owner Surrender']))
    ].copy()

    aggregated_outcomes_filtered = aggregated_outcomes[
        (aggregated_outcomes['animal_type'].isin(['Dog', 'Cat'])) & 
        (aggregated_outcomes['outcome_type'] == 'Adoption')
    ].copy()

    # Aggregate by date
    intake_agg = aggregated_intakes_filtered.groupby('date')['count'].sum().reset_index()
    outcome_agg = aggregated_outcomes_filtered.groupby('date')['count'].sum().reset_index()
    
    # Prepare features
    df_intake_features = prepare_intake_features(intake_agg, intake_col='count', date_col='date')
    df_outcome_features = prepare_intake_features(outcome_agg, intake_col='count', date_col='date')
    
    # Remove rows with NaN values
    df_intake_features = df_intake_features.dropna()
    df_outcome_features = df_outcome_features.dropna()
    
    if prediction_type == "Animal Intakes":
        st.header("🐾 Animal Intake Predictions")
        
        # Initialize and train model
        with st.spinner("Training intake prediction model..."):
            forecaster = XGBoostIntakeForecaster()
            df_sorted, X_train, X_test, y_train, y_test, test_pred = forecaster.fit_and_evaluate(df_intake_features)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            r2 = r2_score(y_test, test_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        
        # Generate future predictions
        with st.spinner(f"Generating {months_ahead} month predictions..."):
            future_dates, future_predictions = forecaster.predict_future(df_intake_features, months_ahead=months_ahead)
        
        # Prepare chart data
        historical_data = df_sorted[['date', 'count']].copy()
        
        # Create future dataframe
        future_df = pd.DataFrame({
            'date': future_dates,
            'count': future_predictions
        })
        
        # Combine historical and future data for chart
        chart_data = pd.concat([
            historical_data.assign(type='Historical'),
            future_df.assign(type='Predicted')
        ], ignore_index=True)
        
        chart_data = chart_data.set_index('date')
        
        # Display chart
        st.subheader(f"Animal Intake Forecast - Next {months_ahead} Months")
        st.line_chart(chart_data['count'])
        
        # Show prediction values
        st.subheader("Predicted Values")
        for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
            st.write(f"**{date.strftime('%B %Y')}**: {pred:.0f} animals")
    
    else:  # Animal Outcomes
        st.header("🏠 Animal Outcome Predictions")
        
        # Initialize and train model
        with st.spinner("Training outcome prediction model..."):
            forecaster = XGBoostOutcomeForecaster()
            df_sorted, X_train, X_test, y_train, y_test, test_pred = forecaster.fit_and_evaluate(df_outcome_features)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            r2 = r2_score(y_test, test_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        
        # Generate future predictions
        with st.spinner(f"Generating {months_ahead} month predictions..."):
            future_dates, future_predictions = forecaster.predict_future(df_outcome_features, months_ahead=months_ahead)
        
        # Prepare chart data
        historical_data = df_sorted[['date', 'count']].copy()
        
        # Create future dataframe
        future_df = pd.DataFrame({
            'date': future_dates,
            'count': future_predictions
        })
        
        # Combine historical and future data for chart
        chart_data = pd.concat([
            historical_data.assign(type='Historical'),
            future_df.assign(type='Predicted')
        ], ignore_index=True)
        
        chart_data = chart_data.set_index('date')
        
        # Display chart
        st.subheader(f"Animal Outcome Forecast - Next {months_ahead} Months")
        st.line_chart(chart_data['count'])
        
        # Show prediction values
        st.subheader("Predicted Values")
        for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
            st.write(f"**{date.strftime('%B %Y')}**: {pred:.0f} animals")
    
    # Additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("- **Algorithm**: XGBoost Regressor")
    st.sidebar.markdown("- **Features**: Seasonal, lag, and rolling window features")
    st.sidebar.markdown("- **Data**: Austin Animal Shelter (Dogs & Cats)")
    
    if prediction_type == "Animal Intakes":
        st.sidebar.markdown("- **Filter**: Stray & Owner Surrender intakes")
    else:
        st.sidebar.markdown("- **Filter**: Adoption outcomes")

if __name__ == "__main__":
    main()