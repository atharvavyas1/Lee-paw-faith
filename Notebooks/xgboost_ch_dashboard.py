import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta

# ========== Define model classes ==========
class XGBoostForecaster:
    def __init__(self, model_type='intake'):
        if model_type == 'intake':
            self.model = xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42
            )
        else:  # outcome
            self.model = xgb.XGBRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                reg_alpha=2.0, reg_lambda=2.0, random_state=42
            )
        self.feature_cols = None

    def prepare_data(self, df, target_col='count', test_size=0.2):
        df_sorted = df.sort_values('date').reset_index(drop=True)
        exclude_cols = [target_col, 'date', 'year']
        self.feature_cols = [col for col in df_sorted.columns if col not in exclude_cols]
        X = df_sorted[self.feature_cols]
        y = df_sorted[target_col]
        return X, y

    def fit(self, df, target_col='count'):
        X, y = self.prepare_data(df, target_col)
        self.model.fit(X, y)

    def forecast_next(self, df_last, forecast_horizon):
        df_forecast = df_last.copy()
        predictions = []
        last_date = pd.to_datetime(df_forecast['date'].iloc[-1])

        for i in range(forecast_horizon):
            new_date = last_date + pd.DateOffset(months=1)
            last_date = new_date

            row = df_forecast.iloc[-1:].copy()

            # Shift lag features
            for lag in [1, 2, 3, 6, 12]:
                if f'count_lag_{lag}' in row.columns:
                    row[f'count_lag_{lag}'] = df_forecast['count'].iloc[-lag]

            # Shift rolling features (keep them same or simple)
            for window in [3, 6, 12]:
                if f'count_rolling_mean_{window}' in row.columns:
                    row[f'count_rolling_mean_{window}'] = df_forecast['count'].tail(window).mean()
                    row[f'count_rolling_std_{window}'] = df_forecast['count'].tail(window).std()
                    row[f'count_rolling_min_{window}'] = df_forecast['count'].tail(window).min()
                    row[f'count_rolling_max_{window}'] = df_forecast['count'].tail(window).max()

            # Seasonal features
            row['month'] = new_date.month
            row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
            row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)
            row['year'] = new_date.year

            # Drop unused columns
            row['date'] = new_date
            row = row[self.feature_cols]
            pred = self.model.predict(row)[0]

            # Add to forecast
            predictions.append({'date': new_date, 'count': pred})

            # Append the prediction to df_forecast for rolling feature updates
            new_row = df_forecast.iloc[-1:].copy()
            new_row['date'] = new_date
            new_row['count'] = pred
            df_forecast = pd.concat([df_forecast, new_row], ignore_index=True)

        return pd.DataFrame(predictions)


# ========== Load and preprocess data ==========
@st.cache_data
def load_data():
    intake_url = "https://data.austintexas.gov/resource/wter-evkm.csv?$limit=500000"
    df_intakes = pd.read_csv(intake_url)
    outcome_url = "https://data.austintexas.gov/resource/9t4d-g238.csv?$limit=500000"
    df_outcomes = pd.read_csv(outcome_url)

    # Filter
    df_intakes = df_intakes[
        (df_intakes['animal_type'].isin(['Dog', 'Cat'])) &
        (df_intakes['intake_type'].isin(['Stray', 'Owner Surrender']))
    ]

    df_outcomes = df_outcomes[
        (df_outcomes['animal_type'].isin(['Dog', 'Cat'])) &
        (df_outcomes['outcome_type'] == 'Adoption')
    ]

    # Aggregate
    intake_agg = df_intakes.groupby('datetime')['animal_id'].count().reset_index()
    outcome_agg = df_outcomes.groupby('datetime')['animal_id'].count().reset_index()
    intake_agg.columns = outcome_agg.columns = ['date', 'count']
    intake_agg['date'] = pd.to_datetime(intake_agg['date'], format='mixed', utc=True).dt.to_period('M').dt.to_timestamp()
    outcome_agg['date'] = pd.to_datetime(outcome_agg['date'], format='mixed', utc=True).dt.to_period('M').dt.to_timestamp()

    # Convert to monthly data
    intake_monthly = intake_agg.groupby('date')['count'].sum().reset_index()
    outcome_monthly = outcome_agg.groupby('date')['count'].sum().reset_index()
    return intake_monthly, outcome_monthly


def add_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year'] = df['date'].dt.year

    for lag in [1, 2, 3, 6, 12]:
        df[f'count_lag_{lag}'] = df['count'].shift(lag)
    for window in [3, 6, 12]:
        df[f'count_rolling_mean_{window}'] = df['count'].rolling(window).mean()
        df[f'count_rolling_std_{window}'] = df['count'].rolling(window).std()
        df[f'count_rolling_min_{window}'] = df['count'].rolling(window).min()
        df[f'count_rolling_max_{window}'] = df['count'].rolling(window).max()

    return df.dropna().reset_index(drop=True)


# ========== Streamlit UI ==========
st.set_page_config(page_title="Austin Animal Shelter Forecast", layout="wide")
st.title("📈 Austin Animal Shelter Forecast Dashboard")

data_choice = st.selectbox("What would you like to predict?", ["Intakes", "Outcomes"])
months_ahead = st.slider("Select months to forecast into the future", 1, 6, 3)

# Load data
intake_df, outcome_df = load_data()

if data_choice == "Intakes":
    df = intake_df.copy()
    model_type = 'intake'
else:
    df = outcome_df.copy()
    model_type = 'outcome'

# Feature engineering
df_features = add_features(df)

# Train model
forecaster = XGBoostForecaster(model_type=model_type)
forecaster.fit(df_features)

# Forecast
forecast_df = forecaster.forecast_next(df_features, months_ahead)

# Combine for chart
chart_df = pd.concat([df[['date', 'count']], forecast_df], ignore_index=True)
chart_df = chart_df.set_index('date')

st.subheader(f"{data_choice} Forecast (Next {months_ahead} months)")
st.line_chart(chart_df)
