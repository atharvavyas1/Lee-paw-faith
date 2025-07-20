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

# ARIMA modeling
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
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
    
    # Fill missing values with 'Unknown'
    intake_processed['animal_type'] = intake_processed['animal_type'].fillna('Unknown')
    intake_processed['intake_type'] = intake_processed['intake_type'].fillna('Unknown')
    outcome_processed['animal_type'] = outcome_processed['animal_type'].fillna('Unknown')
    outcome_processed['outcome_type'] = outcome_processed['outcome_type'].fillna('Unknown')
    
    # Aggregate intake data by month, animal type, and intake type
    intake_monthly = (intake_processed.groupby(['year_month', 'animal_type', 'intake_type'])
                     .size()
                     .reset_index(name='intake_count'))
    
    # Aggregate outcome data by month, animal type, and outcome type
    outcome_monthly = (outcome_processed.groupby(['year_month', 'animal_type', 'outcome_type'])
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

def evaluate_arima_model(X, arima_order):
    """Evaluate ARIMA model and return performance metrics"""
    
    try:
        # Ensure we have enough data
        if len(X) < 20:
            return np.inf, None, f"Insufficient data: {len(X)} observations"
        
        # Check for constant series
        if X.std() == 0 or X.nunique() <= 1:
            return np.inf, None, "Constant series detected"
        
        # Check for missing values
        if X.isna().any():
            X = X.dropna()
            if len(X) < 20:
                return np.inf, None, "Too few observations after removing NaN"
        
        # Split data - use larger training set for better model fitting
        train_size = max(int(len(X) * 0.85), len(X) - 6)  # Leave at least 6 months for testing
        train, test = X[:train_size], X[train_size:]
        
        # Ensure we have enough data for both training and testing
        if len(train) < 15:
            return np.inf, None, f"Insufficient training data: {len(train)} observations"
        
        if len(test) < 1:
            return np.inf, None, "No test data available"
        
        # Fit ARIMA model with error handling
        try:
            model = ARIMA(train, order=arima_order)
            model_fit = model.fit(method_kwargs={'warn_convergence': False})
        except Exception as fit_error:
            return np.inf, None, f"Model fitting failed: {str(fit_error)}"
        
        # Check if model converged properly
        if not hasattr(model_fit, 'params') or model_fit.params is None:
            return np.inf, None, "Model did not converge"
        
        # Make predictions
        try:
            forecast = model_fit.forecast(steps=len(test))
            
            # Check for invalid forecasts
            if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                return np.inf, None, "Invalid forecast values"
            
            # Calculate performance metrics
            mae = mean_absolute_error(test, forecast)
            
            # Calculate MAPE, handling zero values
            test_non_zero = test[test != 0]
            forecast_non_zero = forecast[test != 0]
            
            if len(test_non_zero) > 0:
                mape = mean_absolute_percentage_error(test_non_zero, forecast_non_zero) * 100
            else:
                # If all test values are zero, use normalized MAE
                mape = (mae / (test.mean() + 1e-8)) * 100
            
            # Return MAPE as primary metric
            return mape, model_fit, "Success"
            
        except Exception as forecast_error:
            return np.inf, None, f"Forecasting failed: {str(forecast_error)}"
            
    except Exception as e:
        return np.inf, None, f"General error: {str(e)}"

def auto_arima_grid_search(timeseries):
    """Perform grid search to find best ARIMA parameters"""
    
    st.info("Performing grid search for optimal ARIMA parameters...")
    
    # Define parameter ranges - keep them reasonable
    p_values = range(0, 4)  # AR order
    d_values = range(0, 3)  # Differencing order  
    q_values = range(0, 4)  # MA order
    
    # Generate all combinations
    parameters = list(itertools.product(p_values, d_values, q_values))
    
    best_mape = np.inf
    best_param = None
    best_model = None
    results = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_combinations = len(parameters)
    
    successful_fits = 0
    
    for i, param in enumerate(parameters):
        try:
            # Update progress
            progress_bar.progress((i + 1) / total_combinations)
            status_text.text(f"Testing ARIMA{param} - {i+1}/{total_combinations}")
            
            # Evaluate model
            mape, model, status = evaluate_arima_model(timeseries, param)
            
            # Store results for debugging
            results.append({
                'order': param,
                'mape': mape,
                'status': status
            })
            
            if mape < best_mape and model is not None:
                best_mape = mape
                best_param = param
                best_model = model
                successful_fits += 1
                
        except Exception as e:
            results.append({
                'order': param,
                'mape': np.inf,
                'status': f"Exception: {str(e)}"
            })
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    st.info(f"Tested {total_combinations} parameter combinations")
    st.info(f"Successful model fits: {successful_fits}")
    
    if best_param is not None:
        st.success(f"Best ARIMA order: {best_param}")
        st.success(f"Best MAPE: {best_mape:.2f}%")
        
        # Show top 5 best models
        with st.expander("View Top 5 Best Models"):
            results_df = pd.DataFrame(results)
            results_df = results_df[results_df['mape'] != np.inf].sort_values('mape').head(5)
            st.dataframe(results_df)
        
        return best_model, best_param, best_mape
    else:
        st.error("Could not find suitable model parameters")
        
        # Show some failed attempts for debugging
        with st.expander("Debug Information - Failed Attempts"):
            results_df = pd.DataFrame(results)
            failed_results = results_df[results_df['mape'] == np.inf].head(10)
            st.dataframe(failed_results)
        
        return None, None, np.inf

def create_forecast(model, steps, data_type, animal_type):
    """Create forecast using fitted model"""
    
    try:
        # Generate forecast
        forecast = model.forecast(steps=steps)
        forecast_ci = model.get_forecast(steps=steps).conf_int()
        
        # Create future dates
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=steps, freq='MS')  # MS = Month Start
        
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
    st.set_page_config(page_title="Austin Animal Center ARIMA Analysis", 
                      layout="wide", initial_sidebar_state="expanded")
    
    st.title("🐕 Austin Animal Center ARIMA Time Series Analysis")
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
        
        # Get available animal types and intake/outcome types
        if analysis_type == "Intakes":
            available_animals = list(intake_monthly['animal_type'].unique())
            available_types = list(intake_monthly['intake_type'].unique())
            type_label = "Intake Type"
        else:
            available_animals = list(outcome_monthly['animal_type'].unique())
            available_types = list(outcome_monthly['outcome_type'].unique())
            type_label = "Outcome Type"
        
        selected_animal = st.sidebar.selectbox(
            "Select Animal Type:",
            ["All"] + available_animals
        )
        
        selected_type = st.sidebar.selectbox(
            f"Select {type_label}:",
            ["All"] + available_types
        )
        
        prediction_months = st.sidebar.slider(
            "Prediction Horizon (months):",
            min_value=1, max_value=12, value=6
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
        
        # Type distribution (Intake/Outcome type)
        st.subheader(f"{analysis_type[:-1]} Type Distribution")
        
        if analysis_type == "Intakes":
            type_counts = intake_raw['intake_type'].value_counts()
            type_title = "Intakes by Intake Type"
        else:
            type_counts = outcome_raw['outcome_type'].value_counts()
            type_title = "Outcomes by Outcome Type"
        
        fig_pie_type = px.pie(values=type_counts.values, names=type_counts.index,
                             title=type_title)
        st.plotly_chart(fig_pie_type, use_container_width=True)
    
    with col2:
        st.header("🔧 Model Configuration")
        
        auto_search = st.checkbox("Auto-search optimal parameters", value=True)
        
        if not auto_search:
            st.subheader("Manual ARIMA Parameters")
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
            d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1)  
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
    
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
    
    # Filter by intake/outcome type if not "All"
    type_column = 'intake_type' if data_type == 'intake' else 'outcome_type'
    if selected_type != "All":
        monthly_data = monthly_data[monthly_data[type_column] == selected_type]
        if len(monthly_data) == 0:
            st.warning(f"No {data_type} data available for {selected_animal} - {selected_type}.")
            return
    
    # Create aggregation strategy based on selections
    if selected_animal == "All" and selected_type == "All":
        # Aggregate everything
        monthly_data = monthly_data.groupby('date')[f'{data_type}_count'].sum().reset_index()
        monthly_data['animal_type'] = 'All Animals'
        monthly_data[type_column] = 'All Types'
        analysis_label = 'All Animals - All Types'
    elif selected_animal == "All":
        # Aggregate by type only
        monthly_data = monthly_data.groupby(['date', type_column])[f'{data_type}_count'].sum().reset_index()
        monthly_data['animal_type'] = 'All Animals'
        analysis_label = f'All Animals - {selected_type}'
    elif selected_type == "All":
        # Aggregate by animal only
        monthly_data = monthly_data.groupby(['date', 'animal_type'])[f'{data_type}_count'].sum().reset_index()
        monthly_data[type_column] = 'All Types'
        analysis_label = f'{selected_animal} - All Types'
    else:
        # Both animal and type are specified - no aggregation needed
        analysis_label = f'{selected_animal} - {selected_type}'
    
    # Check minimum data requirements
    min_required_months = 24
    if len(monthly_data) < min_required_months:
        st.warning(f"Insufficient data for reliable ARIMA analysis. Need at least {min_required_months} months of data. Currently have {len(monthly_data)} months.")
        
        # Show available data anyway
        if len(monthly_data) > 0:
            st.subheader("Available Data (Insufficient for ARIMA)")
            chart_data = monthly_data.set_index('date')[f'{data_type}_count']
            st.line_chart(chart_data)
        return
    
    # Create time series
    ts_data = monthly_data.set_index('date')[f'{data_type}_count'].asfreq('MS')
    
    # Fill any missing values with interpolation
    if ts_data.isna().any():
        ts_data = ts_data.interpolate(method='linear')
    
    # Display time series info
    st.subheader("📊 Time Series Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(ts_data))
    with col2:
        st.metric("Mean", f"{ts_data.mean():.1f}")
    with col3:
        st.metric("Std Dev", f"{ts_data.std():.1f}")
    with col4:
        st.metric("Range", f"{ts_data.min():.0f} - {ts_data.max():.0f}")
    
    # Show the time series plot
    st.subheader("Time Series Plot")
    chart_data = pd.DataFrame({f'{analysis_type} Count': ts_data})
    st.line_chart(chart_data)
    
    # Stationarity tests
    st.subheader("🔍 Stationarity Analysis")
    with st.expander("View Stationarity Test Results"):
        is_stationary = check_stationarity(ts_data, f"{analysis_type} for {analysis_label}")
        
        # Plot time series decomposition
        decomposition = plot_decomposition(ts_data, f"{analysis_type} for {analysis_label}")
    
    # Model fitting and forecasting
    st.subheader("🤖 ARIMA Model Training")
    
    if st.button("Train ARIMA Model and Generate Forecast", type="primary"):
        with st.spinner("Training ARIMA model..."):
            
            if auto_search:
                best_model, arima_order, best_mape = auto_arima_grid_search(ts_data)
            else:
                # Use manual parameters
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    best_model = model.fit()
                    arima_order = (p, d, q)
                    
                    # Calculate MAPE on residuals
                    residuals = best_model.resid
                    best_mape = np.mean(np.abs(residuals / ts_data.dropna())) * 100
                    
                    st.success(f"Model trained successfully! MAPE: {best_mape:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return
            
            if best_model is not None:
                # Generate forecast
                forecast, forecast_ci, future_dates = create_forecast(
                    best_model, prediction_months, data_type, analysis_label)
                
                if forecast is not None:
                    # Display model summary
                    st.subheader("📋 Model Summary")
                    
                    # Show key model info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ARIMA Order", f"{arima_order}")
                    with col2:
                        st.metric("AIC", f"{best_model.aic:.2f}")
                    with col3:
                        st.metric("BIC", f"{best_model.bic:.2f}")
                    
                    with st.expander("View Detailed Model Summary"):
                        st.text(str(best_model.summary()))
                    
                    # Plot results
                    st.subheader("📊 Forecast Results")
                    
                    # Prepare data for plotting
                    plot_data = monthly_data.copy()
                    
                    chart_data = plot_time_series_with_forecast(
                        plot_data, forecast, forecast_ci, future_dates,
                        f"{analysis_type} Forecast for {analysis_label}",
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
                        }, index=residuals.index)
                        
                        st.subheader("Residuals Plot")
                        st.line_chart(residual_df)
                        
                        # Residual statistics
                        st.write("**Residual Statistics:**")
                        st.write(f"Mean: {residuals.mean():.4f}")
                        st.write(f"Std: {residuals.std():.4f}")
                        st.write(f"Skewness: {residuals.skew():.4f}")
                        st.write(f"Kurtosis: {residuals.kurtosis():.4f}")
                    
                    with col2:
                        # QQ plot using plotly
                        from scipy import stats
                        qq_data = stats.probplot(residuals.dropna(), dist="norm")
                        
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][1],
                            mode='markers',
                            name='Sample Quantiles',
                            marker=dict(color='blue', size=6)
                        ))
                        fig_qq.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                            mode='lines',
                            name='Theoretical Line',
                            line=dict(color='red', width=2)
                        ))
                        fig_qq.update_layout(
                            title="Q-Q Plot (Normality Check)", 
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles",
                            height=400
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                        
                        # Ljung-Box test for autocorrelation
                        try:
                            ljung_box = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
                            st.write("**Ljung-Box Test (p-values):**")
                            st.write("(p > 0.05 indicates no autocorrelation)")
                            for lag in [1, 5, 10]:
                                if lag <= len(ljung_box):
                                    p_val = ljung_box.iloc[lag-1]['lb_pvalue']
                                    status = "✅" if p_val > 0.05 else "❌"
                                    st.write(f"Lag {lag}: {p_val:.4f} {status}")
                        except:
                            st.write("Could not perform Ljung-Box test")
                    
                    # Performance metrics
                    st.subheader("📊 Model Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MAPE", f"{best_mape:.2f}%")
                    
                    with col2:
                        aic_value = best_model.aic
                        st.metric("AIC", f"{aic_value:.2f}")
                    
                    with col3:
                        bic_value = best_model.bic
                        st.metric("BIC", f"{bic_value:.2f}")
                    
                    with col4:
                        log_likelihood = best_model.llf
                        st.metric("Log-Likelihood", f"{log_likelihood:.2f}")
                    
                    # Additional model insights
                    st.subheader("📈 Model Insights")
                    
                    # Check if differencing was applied
                    if arima_order[1] > 0:
                        st.info(f"The model applied {arima_order[1]} order(s) of differencing to achieve stationarity.")
                    
                    # Interpretation of ARIMA parameters
                    st.write("**Model Interpretation:**")
                    if arima_order[0] > 0:
                        st.write(f"- AR({arima_order[0]}): The model uses the last {arima_order[0]} observation(s) to predict future values")
                    if arima_order[2] > 0:
                        st.write(f"- MA({arima_order[2]}): The model uses the last {arima_order[2]} forecast error(s) to improve predictions")
                    
                    # Forecast interpretation
                    recent_avg = ts_data.tail(6).mean()
                    forecast_avg = forecast.mean()
                    
                    if forecast_avg > recent_avg * 1.05:
                        st.success(f"📈 The model predicts an **increase** in {data_type}s (avg: {forecast_avg:.1f} vs recent avg: {recent_avg:.1f})")
                    elif forecast_avg < recent_avg * 0.95:
                        st.warning(f"📉 The model predicts a **decrease** in {data_type}s (avg: {forecast_avg:.1f} vs recent avg: {recent_avg:.1f})")
                    else:
                        st.info(f"➡️ The model predicts **stable** {data_type}s (avg: {forecast_avg:.1f} vs recent avg: {recent_avg:.1f})")
                else:
                    st.error("Failed to generate forecast")
            else:
                st.error("Could not train ARIMA model with the given data")
    
    # Additional insights
    st.header("💡 Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Patterns")
        
        # Monthly pattern analysis
        if analysis_type == "Intakes":
            pattern_data = intake_daily.copy()
            # Filter by selections if not "All"
            if selected_animal != "All":
                pattern_data = pattern_data[pattern_data['animal_type'] == selected_animal]
            if selected_type != "All":
                pattern_data = pattern_data[pattern_data['intake_type'] == selected_type]
        else:
            pattern_data = outcome_daily.copy()
            # Filter by selections if not "All"
            if selected_animal != "All":
                pattern_data = pattern_data[pattern_data['animal_type'] == selected_animal]
            if selected_type != "All":
                pattern_data = pattern_data[pattern_data['outcome_type'] == selected_type]
        
        if len(pattern_data) > 0:
            pattern_data['month'] = pattern_data['datetime'].dt.month
            monthly_avg = pattern_data.groupby('month').size().reset_index(name='count')
            
            if len(monthly_avg) > 0:
                monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%b')
                
                # Prepare data for streamlit bar chart
                monthly_chart_data = monthly_avg.set_index('month_name')['count']
                
                st.subheader(f"Average Monthly {analysis_type} - {analysis_label}")
                st.bar_chart(monthly_chart_data)
                
                # Find peak and low months
                peak_month = monthly_avg.loc[monthly_avg['count'].idxmax(), 'month_name']
                low_month = monthly_avg.loc[monthly_avg['count'].idxmin(), 'month_name']
                st.write(f"**Peak month:** {peak_month}")
                st.write(f"**Lowest month:** {low_month}")
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
                
                st.subheader(f"Average Weekly {analysis_type} - {analysis_label}")
                st.bar_chart(weekly_chart_data)
                
                # Find peak and low days
                peak_day = weekly_avg.loc[weekly_avg['count'].idxmax(), 'day_of_week']
                low_day = weekly_avg.loc[weekly_avg['count'].idxmin(), 'day_of_week']
                st.write(f"**Peak day:** {peak_day}")
                st.write(f"**Lowest day:** {low_day}")
            else:
                st.info("No weekly pattern data available")
        else:
            st.info("No data available for weekly pattern analysis")
    
    # Trend analysis
    st.subheader("📊 Trend Analysis")
    
    if len(monthly_data) >= 12:
        # Use the filtered monthly_data for trend analysis
        monthly_data_sorted = monthly_data.sort_values('date')
        monthly_data_sorted['year'] = monthly_data_sorted['date'].dt.year
        
        yearly_totals = monthly_data_sorted.groupby('year')[f'{data_type}_count'].sum().reset_index()
        
        if len(yearly_totals) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                latest_year = yearly_totals['year'].max()
                previous_year = latest_year - 1
                
                if previous_year in yearly_totals['year'].values:
                    latest_total = yearly_totals[yearly_totals['year'] == latest_year][f'{data_type}_count'].iloc[0]
                    previous_total = yearly_totals[yearly_totals['year'] == previous_year][f'{data_type}_count'].iloc[0]
                    yoy_change = ((latest_total - previous_total) / previous_total) * 100
                    
                    st.metric(
                        f"Year-over-Year Change ({latest_year})",
                        f"{yoy_change:+.1f}%",
                        delta=f"{latest_total - previous_total:+.0f}"
                    )
            
            with col2:
                # Overall trend (linear regression slope)
                from scipy.stats import linregress
                x_vals = range(len(yearly_totals))
                slope, intercept, r_value, p_value, std_err = linregress(x_vals, yearly_totals[f'{data_type}_count'])
                
                trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
                st.metric("Overall Trend", trend_direction, f"R² = {r_value**2:.3f}")
            
            with col3:
                # Average annual growth rate
                if len(yearly_totals) >= 3:
                    first_year_total = yearly_totals[f'{data_type}_count'].iloc[0]
                    last_year_total = yearly_totals[f'{data_type}_count'].iloc[-1]
                    num_years = len(yearly_totals) - 1
                    
                    if first_year_total > 0:
                        cagr = ((last_year_total / first_year_total) ** (1/num_years) - 1) * 100
                        st.metric("Avg Annual Growth", f"{cagr:+.1f}%")
            
            # Yearly trend chart
            yearly_chart_data = yearly_totals.set_index('year')[f'{data_type}_count']
            st.subheader(f"Annual {analysis_type} Totals - {analysis_label}")
            st.line_chart(yearly_chart_data)
    
    # Data quality information
    st.header("📋 Data Quality Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intake Data Quality")
        intake_quality = {
            "Total Records": len(intake_raw),
            "Missing Animal Types": intake_raw['animal_type'].isna().sum(),
            "Missing Intake Types": intake_raw['intake_type'].isna().sum(),
            "Missing Dates": intake_raw['datetime'].isna().sum(),
            "Unique Animal Types": intake_raw['animal_type'].nunique(),
            "Unique Intake Types": intake_raw['intake_type'].nunique(),
            "Records After Cleaning": len(intake_daily)
        }
        
        for key, value in intake_quality.items():
            st.metric(key, value)
        
        if len(intake_daily) > 0:
            st.write(f"**Date Range:** {intake_daily['datetime'].min().strftime('%Y-%m-%d')} to {intake_daily['datetime'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("Outcome Data Quality")
        outcome_quality = {
            "Total Records": len(outcome_raw),
            "Missing Animal Types": outcome_raw['animal_type'].isna().sum(),
            "Missing Outcome Types": outcome_raw['outcome_type'].isna().sum(),
            "Missing Dates": outcome_raw['datetime'].isna().sum(),
            "Unique Animal Types": outcome_raw['animal_type'].nunique(),
            "Unique Outcome Types": outcome_raw['outcome_type'].nunique(),
            "Records After Cleaning": len(outcome_daily)
        }
        
        for key, value in outcome_quality.items():
            st.metric(key, value)
        
        if len(outcome_daily) > 0:
            st.write(f"**Date Range:** {outcome_daily['datetime'].min().strftime('%Y-%m-%d')} to {outcome_daily['datetime'].max().strftime('%Y-%m-%d')}")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📝 How to Use This Dashboard:
    1. **Select Analysis Type**: Choose between 'Intakes' or 'Outcomes'
    2. **Select Animal Type**: Choose a specific animal type or 'All' for aggregated analysis
    3. **Select Intake/Outcome Type**: Choose a specific type or 'All' for aggregated analysis
    4. **Configure Prediction**: Set the forecast horizon (1-12 months)
    5. **Train Model**: Click the button to automatically find the best ARIMA parameters
    6. **Interpret Results**: Review the forecast, model diagnostics, and insights
    
    ### 🔧 About ARIMA Models:
    - **AR (AutoRegressive)**: Uses past values to predict future values
    - **I (Integrated)**: Applies differencing to make the data stationary
    - **MA (Moving Average)**: Uses past forecast errors to improve predictions
    
    The model automatically searches for the best combination of p, d, q parameters.
    """)

if __name__ == "__main__":
    main()