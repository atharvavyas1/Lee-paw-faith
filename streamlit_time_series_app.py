import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ARIMA/SARIMA modeling
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
import itertools

# ============================================================================
# DATA COLLECTION AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_data():
    """Load and preprocess Austin Animal Center intake and outcome data"""
    
    # Load intake data
    intake_url = "https://data.austintexas.gov/resource/wter-evkm.csv?$limit=500000"
    outcome_url = "https://data.austintexas.gov/resource/9t4d-g238.csv?$limit=500000"
    
    try:
        st.info("Loading intake data...")
        intake_df = pd.read_csv(intake_url)
        
        st.info("Loading outcome data...")
        outcome_df = pd.read_csv(outcome_url)
        
        # Basic validation
        if len(intake_df) == 0:
            st.error("Intake dataset is empty")
            return None, None
            
        if len(outcome_df) == 0:
            st.error("Outcome dataset is empty")
            return None, None
            
        st.success(f"Loaded {len(intake_df)} intake records and {len(outcome_df)} outcome records")
        
        return intake_df, outcome_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("You can try refreshing the page or check your internet connection.")
        return None, None

def preprocess_data(intake_df, outcome_df):
    """Preprocess the intake and outcome data for time series analysis"""
    
    # Process intake data
    intake_processed = intake_df.copy()
    try:
        intake_processed['datetime'] = pd.to_datetime(intake_processed['datetime'], errors='coerce')
    except:
        st.error("Error parsing intake datetime. Trying alternative formats...")
        intake_processed['datetime'] = pd.to_datetime(intake_processed['datetime'], 
                                                    infer_datetime_format=True, errors='coerce')
    
    # Remove rows with invalid dates
    intake_processed = intake_processed.dropna(subset=['datetime'])
    intake_processed['date'] = intake_processed['datetime'].dt.date
    intake_processed['year_month'] = intake_processed['datetime'].dt.to_period('M')
    
    # Process outcome data  
    outcome_processed = outcome_df.copy()
    try:
        outcome_processed['datetime'] = pd.to_datetime(outcome_processed['datetime'], errors='coerce')
    except:
        st.error("Error parsing outcome datetime. Trying alternative formats...")
        outcome_processed['datetime'] = pd.to_datetime(outcome_processed['datetime'], 
                                                     infer_datetime_format=True, errors='coerce')
    
    # Remove rows with invalid dates
    outcome_processed = outcome_processed.dropna(subset=['datetime'])
    outcome_processed['date'] = outcome_processed['datetime'].dt.date
    outcome_processed['year_month'] = outcome_processed['datetime'].dt.to_period('M')
    
    # Check if we have enough data after cleaning
    if len(intake_processed) == 0:
        st.error("No valid intake data after datetime parsing")
        return None, None, None, None
    
    if len(outcome_processed) == 0:
        st.error("No valid outcome data after datetime parsing")
        return None, None, None, None
    
    # Fill missing animal_type with 'Unknown'
    intake_processed['animal_type'] = intake_processed['animal_type'].fillna('Unknown')
    outcome_processed['animal_type'] = outcome_processed['animal_type'].fillna('Unknown')
    
    # Aggregate intake data by month and animal type
    intake_monthly = (intake_processed.groupby(['year_month', 'animal_type'])
                     .size()
                     .reset_index(name='intake_count'))
    
    # Aggregate outcome data by month and animal type
    outcome_monthly = (outcome_processed.groupby(['year_month', 'animal_type'])
                      .size()
                      .reset_index(name='outcome_count'))
    
    # Convert period to datetime for plotting
    intake_monthly['date'] = intake_monthly['year_month'].dt.to_timestamp()
    outcome_monthly['date'] = outcome_monthly['year_month'].dt.to_timestamp()
    
    return intake_monthly, outcome_monthly, intake_processed, outcome_processed

# ============================================================================
# TIME SERIES ANALYSIS FUNCTIONS
# ============================================================================

def check_stationarity(timeseries, title):
    """Check stationarity using Augmented Dickey-Fuller test"""
    
    # Drop NaN values and convert to numpy array to handle edge cases
    clean_series = timeseries.dropna()
    
    # Check if series is empty
    if len(clean_series) == 0:
        st.error(f"No valid data for stationarity test: {title}")
        return False
    
    # Convert to numeric and drop any remaining non-numeric values
    try:
        clean_series = pd.to_numeric(clean_series, errors='coerce').dropna()
    except:
        st.error(f"Cannot convert series to numeric values: {title}")
        return False
    
    # Check again after numeric conversion
    if len(clean_series) == 0:
        st.error(f"No valid numeric data for stationarity test: {title}")
        return False
    
    # Check if series is constant (all values are the same)
    if clean_series.nunique() <= 1 or clean_series.std() == 0:
        st.warning(f"Series is constant (no variation): {title}")
        st.write("Cannot perform ADF test on constant series")
        return True  # Constant series is technically stationary
    
    # Check if series has sufficient length
    if len(clean_series) < 12:
        st.warning(f"Insufficient data for reliable stationarity test: {title} (only {len(clean_series)} observations)")
        return False
    
    try:
        # Perform Augmented Dickey-Fuller test
        result = adfuller(clean_series)
        
        st.subheader(f'Stationarity Test Results for {title}')
        st.write(f'ADF Statistic: {result[0]:.6f}')
        st.write(f'p-value: {result[1]:.6f}')
        st.write('Critical Values:')
        for key, value in result[4].items():
            st.write(f'\t{key}: {value:.6f}')
        
        if result[1] <= 0.05:
            st.success("Series is stationary (reject null hypothesis)")
            return True
        else:
            st.warning("Series is non-stationary (fail to reject null hypothesis)")
            return False
            
    except Exception as e:
        st.error(f"Error performing stationarity test: {str(e)}")
        st.write(f"Series info: length={len(clean_series)}, min={clean_series.min()}, max={clean_series.max()}")
        return False

def plot_decomposition(timeseries, title, period=12):
    """Plot time series decomposition using streamlit charts"""
    
    try:
        decomposition = seasonal_decompose(timeseries.dropna(), model='additive', period=period)
        
        st.subheader(f"Time Series Decomposition: {title}")
        
        # Create individual streamlit line charts for each component
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Series**")
            original_df = pd.DataFrame({'Original': decomposition.observed})
            st.line_chart(original_df)
            
            st.write("**Seasonal Component**")
            seasonal_df = pd.DataFrame({'Seasonal': decomposition.seasonal})
            st.line_chart(seasonal_df)
        
        with col2:
            st.write("**Trend Component**")
            trend_df = pd.DataFrame({'Trend': decomposition.trend})
            st.line_chart(trend_df)
            
            st.write("**Residual Component**")
            residual_df = pd.DataFrame({'Residual': decomposition.resid})
            st.line_chart(residual_df)
        
        return decomposition
    except Exception as e:
        st.error(f"Could not perform decomposition: {e}")
        return None

def evaluate_arima_model(X, arima_order, seasonal_order=None):
    """Evaluate ARIMA/SARIMA model and return MAPE"""
    
    try:
        # Check if series has sufficient data
        if len(X) < 24:
            return np.inf, None
        
        # Check if series is constant
        if X.nunique() <= 1:
            return np.inf, None
        
        # Split data into train and test
        train_size = int(len(X) * 0.8)
        train, test = X[:train_size], X[train_size:]
        
        # Ensure minimum sizes
        if len(train) < 12 or len(test) < 1:
            return np.inf, None
        
        # Fit model
        if seasonal_order:
            model = SARIMAX(train, order=arima_order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(train, order=arima_order)
        
        model_fit = model.fit(disp=False)
        
        # Make predictions
        forecast = model_fit.forecast(steps=len(test))
        
        # Check for invalid forecasts
        if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
            return np.inf, None
        
        # Calculate MAPE only if test values are not zero
        if np.any(test == 0):
            # Use MAE instead of MAPE when there are zero values
            mae = np.mean(np.abs(test - forecast))
            mape = mae / np.mean(np.abs(test)) * 100 if np.mean(np.abs(test)) > 0 else np.inf
        else:
            mape = mean_absolute_percentage_error(test, forecast) * 100
        
        return mape, model_fit
        
    except Exception as e:
        return np.inf, None

def auto_arima_grid_search(timeseries, seasonal=False):
    """Perform grid search to find best ARIMA/SARIMA parameters"""
    
    st.info("Performing grid search for optimal ARIMA parameters...")
    
    # Define parameter ranges
    p = d = q = range(0, 3)
    
    if seasonal:
        # For SARIMA
        P = D = Q = range(0, 2)
        s = [12]  # Monthly seasonality
        
        parameters = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) 
                     for x in itertools.product(p, d, q, P, D, Q, s)]
        
        best_mape = np.inf
        best_param = None
        best_model = None
        
        progress_bar = st.progress(0)
        total_combinations = len(parameters)
        
        for i, param in enumerate(parameters):
            try:
                arima_order = (param[0], param[1], param[2])
                seasonal_order = (param[3], param[4], param[5], param[6])
                
                mape, model = evaluate_arima_model(timeseries, arima_order, seasonal_order)
                
                if mape < best_mape:
                    best_mape = mape
                    best_param = param
                    best_model = model
                    
            except:
                continue
                
            progress_bar.progress((i + 1) / total_combinations)
        
        progress_bar.empty()
        
        if best_param:
            arima_order = (best_param[0], best_param[1], best_param[2])
            seasonal_order = (best_param[3], best_param[4], best_param[5], best_param[6])
            st.success(f"Best SARIMA order: {arima_order} x {seasonal_order}")
            st.success(f"Best MAPE: {best_mape:.2f}%")
            return best_model, arima_order, seasonal_order, best_mape
        
    else:
        # For ARIMA
        parameters = list(itertools.product(p, d, q))
        
        best_mape = np.inf
        best_param = None
        best_model = None
        
        progress_bar = st.progress(0)
        total_combinations = len(parameters)
        
        for i, param in enumerate(parameters):
            try:
                mape, model = evaluate_arima_model(timeseries, param)
                
                if mape < best_mape:
                    best_mape = mape
                    best_param = param
                    best_model = model
                    
            except:
                continue
                
            progress_bar.progress((i + 1) / total_combinations)
        
        progress_bar.empty()
        
        if best_param:
            st.success(f"Best ARIMA order: {best_param}")
            st.success(f"Best MAPE: {best_mape:.2f}%")
            return best_model, best_param, None, best_mape
    
    st.error("Could not find suitable model parameters")
    return None, None, None, np.inf

def create_forecast(model, steps, data_type, animal_type):
    """Create forecast using fitted model"""
    
    try:
        # Generate forecast
        forecast = model.forecast(steps=steps)
        forecast_ci = model.get_forecast(steps=steps).conf_int()
        
        # Create future dates
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=steps, freq='M')
        
        return forecast, forecast_ci, future_dates
    except Exception as e:
        st.error(f"Error creating forecast: {e}")
        return None, None, None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_time_series_with_forecast(data, forecast, forecast_ci, future_dates, 
                                 title, data_type, date_range=None):
    """Plot time series data with forecast using Streamlit line chart"""
    
    # Filter data based on date range if provided
    if date_range:
        mask = (data['date'] >= pd.to_datetime(date_range[0])) & (data['date'] <= pd.to_datetime(date_range[1]))
        plot_data = data[mask]
    else:
        plot_data = data
    
    count_col = f'{data_type}_count'
    
    # Prepare historical data
    if 'animal_type' in plot_data.columns and len(plot_data['animal_type'].unique()) > 1:
        # Multiple animal types - pivot to have each as a column
        historical_chart = plot_data.pivot_table(
            index='date', 
            columns='animal_type', 
            values=count_col, 
            fill_value=0
        )
    else:
        # Single animal type or aggregated data
        historical_chart = plot_data.set_index('date')[[count_col]]
        if 'animal_type' in plot_data.columns:
            animal_name = plot_data['animal_type'].iloc[0] if len(plot_data) > 0 else 'Data'
        else:
            animal_name = 'All Animals'
        historical_chart.columns = [f'{animal_name} (Historical)']
    
    # Combine with forecast if available
    if forecast is not None and len(forecast) > 0:
        # Create forecast dataframe
        forecast_df = pd.DataFrame(index=future_dates)
        
        if len(historical_chart.columns) == 1:
            # Single series
            forecast_df[f'{historical_chart.columns[0].split(" (")[0]} (Forecast)'] = forecast
            
            # Add confidence intervals if available
            if forecast_ci is not None:
                forecast_df[f'{historical_chart.columns[0].split(" (")[0]} (Lower Bound)'] = forecast_ci.iloc[:, 0]
                forecast_df[f'{historical_chart.columns[0].split(" (")[0]} (Upper Bound)'] = forecast_ci.iloc[:, 1]
        else:
            # Multiple series - forecast only for the selected animal type
            forecast_df['Forecast'] = forecast
            if forecast_ci is not None:
                forecast_df['Lower Bound'] = forecast_ci.iloc[:, 0]
                forecast_df['Upper Bound'] = forecast_ci.iloc[:, 1]
        
        # Combine historical and forecast data
        combined_data = pd.concat([historical_chart, forecast_df], sort=True)
    else:
        combined_data = historical_chart
    
    st.subheader(title)
    st.line_chart(combined_data)
    
    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(historical_chart) > 0:
            avg_historical = historical_chart.mean().iloc[0] if len(historical_chart.columns) == 1 else historical_chart.sum(axis=1).mean()
            st.metric("Avg Historical", f"{avg_historical:.1f}")
    
    with col2:
        if forecast is not None and len(forecast) > 0:
            avg_forecast = forecast.mean()
            st.metric("Avg Forecast", f"{avg_forecast:.1f}")
    
    with col3:
        if forecast is not None and len(forecast) > 0 and len(historical_chart) > 0:
            historical_mean = historical_chart.mean().iloc[0] if len(historical_chart.columns) == 1 else historical_chart.sum(axis=1).mean()
            change_pct = ((forecast.mean() - historical_mean) / historical_mean) * 100
            st.metric("Forecast Change", f"{change_pct:+.1f}%")
    
    return combined_data

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

def main():
    st.set_page_config(page_title="Austin Animal Center Time Series Analysis", 
                      layout="wide", initial_sidebar_state="expanded")
    
    st.title("🐕 Austin Animal Center Time Series Analysis Dashboard")
    st.markdown("*Predicting animal intake and outcome patterns using ARIMA modeling*")
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Load data
    with st.spinner("Loading data..."):
        intake_raw, outcome_raw = load_data()
        
    if intake_raw is None or outcome_raw is None:
        st.error("Failed to load data. Please check your internet connection.")
        return
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        result = preprocess_data(intake_raw, outcome_raw)
        
    if result[0] is None:  # Check if preprocessing failed
        st.error("Data preprocessing failed. Please check the data quality.")
        return
    
    intake_monthly, outcome_monthly, intake_daily, outcome_daily = result
    
    # Sidebar controls
    if len(intake_monthly) > 0:
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type:",
            ["Intakes", "Outcomes"]
        )
        
        # Get available animal types
        if analysis_type == "Intakes":
            available_animals = list(intake_monthly['animal_type'].unique())
        else:
            available_animals = list(outcome_monthly['animal_type'].unique())
        
        selected_animal = st.sidebar.selectbox(
            "Select Animal Type:",
            ["All"] + available_animals
        )
        
        prediction_months = st.sidebar.slider(
            "Prediction Horizon (months):",
            min_value=1, max_value=6, value=3
        )
        
        # Date range slider for visualization
        min_date = min(intake_monthly['date'].min(), outcome_monthly['date'].min())
        max_date = max(intake_monthly['date'].max(), outcome_monthly['date'].max())
        
        date_range = st.sidebar.date_input(
            "Date Range for Visualization:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        st.error("No monthly data available for analysis")
        return
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Overview")
        
        # Display basic statistics
        st.subheader("Dataset Summary")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Total Intake Records", len(intake_raw))
            if len(intake_daily) > 0:
                st.metric("Intake Date Range", 
                         f"{intake_daily['datetime'].min().strftime('%Y-%m-%d')} to {intake_daily['datetime'].max().strftime('%Y-%m-%d')}")
            else:
                st.metric("Intake Date Range", "No valid dates")
        
        with col_b:
            st.metric("Total Outcome Records", len(outcome_raw))
            if len(outcome_daily) > 0:
                st.metric("Outcome Date Range",
                         f"{outcome_daily['datetime'].min().strftime('%Y-%m-%d')} to {outcome_daily['datetime'].max().strftime('%Y-%m-%d')}")
            else:
                st.metric("Outcome Date Range", "No valid dates")
        
        # Animal type distribution
        st.subheader("Animal Type Distribution")
        
        if analysis_type == "Intakes":
            animal_counts = intake_raw['animal_type'].value_counts()
        else:
            animal_counts = outcome_raw['animal_type'].value_counts()
        
        fig_pie = px.pie(values=animal_counts.values, names=animal_counts.index,
                        title=f"{analysis_type} by Animal Type")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.header("🔧 Model Configuration")
        
        use_seasonal = st.checkbox("Use Seasonal ARIMA (SARIMA)", value=True)
        auto_search = st.checkbox("Auto-search optimal parameters", value=True)
        
        if not auto_search:
            st.subheader("Manual ARIMA Parameters")
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
            d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1)  
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
            
            if use_seasonal:
                P = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
                D = st.number_input("Seasonal Differencing (D)", min_value=0, max_value=1, value=1)
                Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)
                s = st.number_input("Seasonal period (s)", min_value=1, max_value=12, value=12)
    
    # Time series analysis
    st.header("📈 Time Series Analysis")
    
    # Prepare data for analysis
    if analysis_type == "Intakes":
        monthly_data = intake_monthly.copy()
        data_type = "intake"
    else:
        monthly_data = outcome_monthly.copy()
        data_type = "outcome"
    
    # Check if we have data for the selected analysis type
    if len(monthly_data) == 0:
        st.warning(f"No {data_type} data available for analysis.")
        return
    
    # Filter by animal type if not "All"
    if selected_animal != "All":
        monthly_data = monthly_data[monthly_data['animal_type'] == selected_animal]
        if len(monthly_data) == 0:
            st.warning(f"No {data_type} data available for {selected_animal}.")
            return
    else:
        # Aggregate all animal types
        monthly_data = monthly_data.groupby('date')[f'{data_type}_count'].sum().reset_index()
        monthly_data['animal_type'] = 'All Animals'
    
    if len(monthly_data) < 24:  # Need minimum data for analysis
        st.warning(f"Insufficient data for time series analysis. Need at least 24 months of data. Currently have {len(monthly_data)} months.")
        
        # Show available data anyway
        if len(monthly_data) > 0:
            st.subheader("Available Data (Insufficient for ARIMA)")
            chart_data = monthly_data.set_index('date')[f'{data_type}_count']
            st.line_chart(chart_data)
        return
    
    # Create time series
    ts_data = monthly_data.set_index('date')[f'{data_type}_count'].asfreq('M')
    
    # Stationarity tests
    st.subheader("🔍 Stationarity Analysis")
    with st.expander("View Stationarity Test Results"):
        is_stationary = check_stationarity(ts_data, f"{analysis_type} for {selected_animal}")
        
        # Plot time series decomposition
        decomposition = plot_decomposition(ts_data, f"{analysis_type} for {selected_animal}")
    
    # Model fitting and forecasting
    st.subheader("🤖 ARIMA Model Training")
    
    if st.button("Train Model and Generate Forecast", type="primary"):
        with st.spinner("Training ARIMA model..."):
            
            if auto_search:
                best_model, arima_order, seasonal_order, best_mape = auto_arima_grid_search(
                    ts_data, seasonal=use_seasonal)
            else:
                # Use manual parameters
                try:
                    if use_seasonal:
                        model = SARIMAX(ts_data, order=(p, d, q), 
                                      seasonal_order=(P, D, Q, s))
                    else:
                        model = ARIMA(ts_data, order=(p, d, q))
                    
                    best_model = model.fit(disp=False)
                    arima_order = (p, d, q)
                    seasonal_order = (P, D, Q, s) if use_seasonal else None
                    
                    # Calculate MAPE on train data
                    residuals = best_model.resid
                    best_mape = np.mean(np.abs(residuals / ts_data.dropna())) * 100
                    
                    st.success(f"Model trained successfully! MAPE: {best_mape:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return
            
            if best_model is not None:
                # Generate forecast
                forecast, forecast_ci, future_dates = create_forecast(
                    best_model, prediction_months, data_type, selected_animal)
                
                if forecast is not None:
                    # Display model summary
                    st.subheader("📋 Model Summary")
                    with st.expander("View Model Details"):
                        st.text(str(best_model.summary()))
                    
                    # Plot results
                    st.subheader("📊 Forecast Results")
                    
                    # Prepare data for plotting
                    plot_data = monthly_data.copy()
                    
                    chart_data = plot_time_series_with_forecast(
                        plot_data, forecast, forecast_ci, future_dates,
                        f"{analysis_type} Forecast for {selected_animal}",
                        data_type, date_range
                    )
                    
                    # Display forecast values
                    st.subheader("🔮 Forecast Values")
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Count': forecast.round(0).astype(int),
                        'Lower Bound': forecast_ci.iloc[:, 0].round(0).astype(int) if forecast_ci is not None else None,
                        'Upper Bound': forecast_ci.iloc[:, 1].round(0).astype(int) if forecast_ci is not None else None
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Model diagnostics
                    st.subheader("🔧 Model Diagnostics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Residual plot using streamlit line chart
                        residuals = best_model.resid
                        residual_df = pd.DataFrame({
                            'Residuals': residuals
                        }, index=range(len(residuals)))
                        
                        st.subheader("Residuals Plot")
                        st.line_chart(residual_df)
                    
                    with col2:
                        # QQ plot using plotly (since st.line_chart doesn't support scatter plots well)
                        from scipy import stats
                        qq_data = stats.probplot(residuals.dropna(), dist="norm")
                        
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][1],
                            mode='markers',
                            name='Sample Quantiles'
                        ))
                        fig_qq.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                            mode='lines',
                            name='Theoretical Line'
                        ))
                        fig_qq.update_layout(title="Q-Q Plot", 
                                           xaxis_title="Theoretical Quantiles",
                                           yaxis_title="Sample Quantiles")
                        st.plotly_chart(fig_qq, use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("📊 Model Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MAPE", f"{best_mape:.2f}%")
                    
                    with col2:
                        aic_value = best_model.aic
                        st.metric("AIC", f"{aic_value:.2f}")
                    
                    with col3:
                        bic_value = best_model.bic
                        st.metric("BIC", f"{bic_value:.2f}")
    
    # Additional insights
    st.header("💡 Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Patterns")
        
        # Monthly pattern analysis
        if analysis_type == "Intakes":
            pattern_data = intake_daily.copy()
        else:
            pattern_data = outcome_daily.copy()
        
        if len(pattern_data) > 0:
            pattern_data['month'] = pattern_data['datetime'].dt.month
            monthly_avg = pattern_data.groupby('month').size().reset_index(name='count')
            
            if len(monthly_avg) > 0:
                monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%b')
                
                # Prepare data for streamlit bar chart
                monthly_chart_data = monthly_avg.set_index('month_name')['count']
                
                st.subheader(f"Average Monthly {analysis_type}")
                st.bar_chart(monthly_chart_data)
            else:
                st.info("No monthly pattern data available")
        else:
            st.info("No data available for seasonal pattern analysis")
    
    with col2:
        st.subheader("Weekly Patterns")
        
        # Weekly pattern analysis
        if len(pattern_data) > 0:
            pattern_data['day_of_week'] = pattern_data['datetime'].dt.day_name()
            weekly_avg = pattern_data.groupby('day_of_week').size().reset_index(name='count')
            
            if len(weekly_avg) > 0:
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_avg['day_of_week'] = pd.Categorical(weekly_avg['day_of_week'], categories=day_order, ordered=True)
                weekly_avg = weekly_avg.sort_values('day_of_week')
                
                # Prepare data for streamlit bar chart
                weekly_chart_data = weekly_avg.set_index('day_of_week')['count']
                
                st.subheader(f"Average Weekly {analysis_type}")
                st.bar_chart(weekly_chart_data)
            else:
                st.info("No weekly pattern data available")
        else:
            st.info("No data available for weekly pattern analysis")
    
    # Data quality information
    st.header("📋 Data Quality Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intake Data Quality")
        intake_quality = {
            "Total Records": len(intake_raw),
            "Missing Animal Types": intake_raw['animal_type'].isna().sum(),
            "Missing Dates": intake_raw['datetime'].isna().sum(),
            "Date Range": f"{intake_daily['datetime'].min().strftime('%Y-%m-%d')} to {intake_daily['datetime'].max().strftime('%Y-%m-%d')}",
            "Unique Animal Types": intake_raw['animal_type'].nunique()
        }
        
        for key, value in intake_quality.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("Outcome Data Quality")
        outcome_quality = {
            "Total Records": len(outcome_raw),
            "Missing Animal Types": outcome_raw['animal_type'].isna().sum(),
            "Missing Dates": outcome_raw['datetime'].isna().sum(),
            "Date Range": f"{outcome_daily['datetime'].min().strftime('%Y-%m-%d')} to {outcome_daily['datetime'].max().strftime('%Y-%m-%d')}",
            "Unique Animal Types": outcome_raw['animal_type'].nunique()
        }
        
        for key, value in outcome_quality.items():
            st.metric(key, value)

if __name__ == "__main__":
    main()