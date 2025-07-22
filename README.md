# 🐾 Lee-paw-faith: Animal Shelter Analytics & Prediction Platform🐾

A comprehensive machine learning and analytics platform designed to help animal shelters worldwide optimize their operations and improve adoption outcomes through data-driven insights.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Features & Engineering](#features--engineering)
- [Machine Learning Models](#machine-learning-models)
- [Dashboards & Applications](#dashboards--applications)
- [API Deployment](#api-deployment)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Technical Architecture](#technical-architecture)
- [Contributing](#contributing)

## 🎯 Project Overview

Lee-paw-faith is a comprehensive analytics platform that leverages machine learning to help animal shelters:
- **Predict adoption likelihood** for incoming animals
- **Identify patterns** in shelter operations
- **Optimize resource allocation** through time series forecasting
- **Detect anomalies** in shelter data
- **Visualize shelter performance** through interactive dashboards

The project analyzes data from multiple animal shelters including Austin Animal Center, Long Beach Animal Shelter, and Sonoma County Animal Shelter.

## 📊 Data Sources

### Austin Animal Center Data

The primary dataset comes from Austin Animal Center, containing comprehensive intake and outcome records.

#### Original Features

**Intakes Data**

| Feature Category      | Features              | Description                                 |
|----------------------|----------------------|---------------------------------------------|
| Identification       | `animal_id`            | Unique animal identifier                    |
| Identification       | `name`                 | Name of the animal (if available)           |
| Temporal             | `datetime`             | Date and time of intake                     |
| Temporal             | `datetime2`            | Month and year of intake                    |
| Location             | `found_location`       | Location where animal was found             |
| Operational          | `intake_type`          | Type of intake (e.g., Stray, Owner Surrender) |
| Condition            | `intake_condition`     | Condition of the animal at intake           |
| Animal Information   | `animal_type`          | Animal type (e.g., Dog, Cat)                |
| Animal Information   | `sex_upon_intake`      | Sex of the animal at intake                 |
| Animal Information   | `age_upon_intake`      | Age of the animal at intake                 |
| Animal Information   | `breed`                | Animal breed                                |
| Animal Information   | `color`                | Animal color                                |

**Outcomes Data**

| Feature Category      | Features              | Description                                 |
|----------------------|----------------------|---------------------------------------------|
| Identification       | `animal_id`            | Unique animal identifier                    |
| Demographic          | `date_of_birth`        | Date of birth of the animal                 |
| Identification       | `name`                 | Name of the animal (if available)           |
| Temporal             | `datetime`             | Date and time of outcome                    |
| Temporal             | `monthyear`            | Month and year of outcome                   |
| Operational          | `outcome_type`         | Type of outcome (e.g., Adoption, Euthanasia)|
| Operational          | `outcome_subtype`      | Subtype of outcome                          |
| Animal Information   | `animal_type`          | Animal type (e.g., Dog, Cat)                |
| Animal Information   | `sex_upon_outcome`     | Sex of the animal at outcome                |
| Animal Information   | `age_upon_outcome`     | Age of the animal at outcome                |
| Animal Information   | `breed`                | Animal breed                                |
| Animal Information   | `color`                | Animal color                                |

#### Engineered Features

| Feature Category | Features | Description |
|-----------------|----------|-------------|
| **Breed Analysis** | `is_mixed`, `num_breeds`, `primary_breed`, `breed_complexity` | Breed classification and complexity |
| **Size Categories** | `size_category` | Categorized size (Toy, Small, Medium, Large) |
| **Breed Groups** | `is_working`, `is_sporting`, `is_terrier`, `is_popular_dog`, `is_popular_cat` | Functional breed classifications |
| **Geographic** | `is_austin_metro`, `is_core_austin`, `is_travis_county`, `distance_category` | Location-based features |
| **Visit History** | `visit_count`, `is_return_visit`, `is_frequent_returner`, `days_since_last_visit` | Return visit patterns |
| **Temporal** | `month_intake`, `intake_year`, `intake_day`, `intake_dayofweek`, `intake_season`, `intake_is_weekend` | Time-based features |
| **Demographic** | `has_name`, `age_upon_intake_days`, `birth_year`, `birth_month` | Animal demographics |

### Long Beach Animal Shelter Data

Secondary dataset from Long Beach Animal Shelter used for comparative analysis and validation.

### Sonoma County Animal Shelter Data

Additional dataset used for the Local Urban Companion Archive (LUCA) dashboard.

## 🤖 Machine Learning Models

### Austin Animal Center - H2O AutoML Stacked Ensemble

#### Model Performance
- **Accuracy**: 78% (baseline)
- **Algorithm**: Stacked Ensemble (XGBoost + GBM + DRF)
- **Training Time**: 4 hours (14,400 seconds)
- **Features**: 40+ engineered features
- **Target Variable**: `Is_adopted` (binary classification)

#### Model Architecture
```python
# H2O AutoML Configuration
aml = H2OAutoML(
    max_models=50,
    max_runtime_secs=14400,
    include_algos=["XGBoost", "GBM", "DRF", "StackedEnsemble"],
    seed=42,
    balance_classes=True,
    sort_metric="AUC"
)
```

#### Hyperparameter Tuning Recommendations

The current model achieves 78% accuracy, but there's significant room for improvement through:

1. **Feature Engineering**:
   - Create interaction features between breed and age
   - Develop seasonal adoption patterns
   - Engineer location-based features

2. **Model Optimization**:
   - Grid search for optimal hyperparameters
   - Ensemble different algorithms
   - Cross-validation strategies

3. **Data Quality**:
   - Handle missing values more sophisticatedly
   - Feature selection techniques
   - Outlier detection and treatment

### Anomaly Detection - Isolation Forest

#### Austin Data Results
- **Outlier Detection**: Successfully identified anomalous patterns
- **Key Findings**: 
  - Animals with extreme visit counts (15+ visits)
  - Unusual age patterns (very young or very old animals)
  - Geographic outliers (animals from distant locations)

#### Long Beach Data Results
- **Outlier Detection**: Identified similar patterns to Austin
- **Performance**: Consistent with Austin findings

### Clustering Analysis - Spectral Clustering

#### Long Beach EDA Results
- **Best Silhouette Score**: 0.4204 (K=3 clusters)
- **Cluster Interpretation**:
  - Cluster 1: Young, healthy animals with high adoption rates
  - Cluster 2: Older animals with medical conditions
  - Cluster 3: Mixed characteristics with moderate outcomes

#### Performance Comparison
| Algorithm | Optimal K | Silhouette Score | Performance |
|-----------|-----------|------------------|-------------|
| K-Means | 8 | 0.1087 | Good |
| Agglomerative | 2 | 0.1948 | Best |
| DBSCAN | 18 | -0.2439 | Poor |
| **Spectral Clustering** | **3** | **0.4204** | **Best** |

### Long Beach ML Model - Abandoned Pursuit

The Long Beach ML model was developed but ultimately abandoned due to poor performance:

#### Results
- **Test Accuracy**: 65.11%
- **Mean Per-Class Accuracy**: 64.00%
- **Cross-Validation Accuracy**: 63.41%

#### Reasons for Abandonment
1. **Low Accuracy**: Significantly lower than Austin model (78% vs 65%)
2. **Data Quality Issues**: Inconsistent data patterns
3. **Feature Limitations**: Fewer engineered features available
4. **Resource Allocation**: Focus shifted to improving Austin model

## 📈 Dashboards & Applications

### 1. Main Dashboard (`main_dashboard.py`)
**Purpose**: Comprehensive analytics dashboard for Austin Animal Center

**Live Demo**: [🐾 Austin Animal Shelter Analytics Dashboard](https://lee-paw-faith-5dh5nq5c2gzfwsdrlvm8t5.streamlit.app/)

**Features**:
- Real-time data visualization
- Adoption prediction interface
- Anomaly detection results
- Performance metrics
- Interactive charts and graphs

**Technologies**: Streamlit, Plotly, Pandas

### 2. Local Urban Companion Archive - LUCA (`streamlit_app_v3.py`)
**Purpose**: Geographic visualization of Sonoma County Animal Shelter outcomes

**Live Demo**: [🗺️ Local Urban Companion Archive (LUCA)](https://lee-paw-faith-hmqkeu7aa3ollusevhcugz.streamlit.app/)

**Features**:
- Interactive map with animal locations
- Color-coded outcome types
- Filtering by animal type and outcome
- Real-time data loading from API
- Performance-optimized rendering

**Technologies**: Streamlit, PyDeck, Mapbox

### 3. Time Series Analysis (`xgboost_ch_dashboard.py`)
**Purpose**: XGBoost-based forecasting for shelter intake and outcome predictions

**Live Demo**: [📊 Time Series Forecasting Dashboard](https://lee-paw-faith-cnmvbn77ecy9st7swszfcs.streamlit.app/)

**Features**:
- XGBoost regression models for intake/outcome forecasting
- Advanced feature engineering with lag and rolling statistics
- Seasonal pattern recognition using trigonometric features
- Interactive forecasting interface (1-6 months ahead)
- Real-time model training and prediction
- Visual comparison of actual vs forecasted values

**Technologies**: Streamlit, XGBoost, Altair, Pandas, NumPy

## 🚀 API Deployment

### FastAPI Service (`main.py`)

The Austin Animal Shelter Stacked Ensemble model is deployed as a REST API using FastAPI.

#### API Endpoints

```python
# Health Check
GET /health
# Returns: {"status": "healthy", "model_loaded": true}

# Prediction Endpoint
POST /predict
# Input: AnimalFeatures schema
# Output: PredictionResponse with adoption probability
```

#### Input Schema
```python
class AnimalFeatures(BaseModel):
    animal_type_intake: str
    sex_upon_intake: str
    age_upon_intake_days: float
    intake_type: str
    intake_condition: str
    # ... 40+ additional features
```

#### Sample Input
```json
{
 "animal_type_intake": "Cat",
 "sex_upon_intake": "Spayed Female",
 "age_upon_intake_days": 180,
 "intake_type": "Owner Surrender",
 "intake_condition": "Normal",
 "has_name": 1,
 "is_mixed": 0,
 "num_breeds": 1,
 "size_category": "Small",
 "is_working": 0,
 "is_sporting": 0,
 "is_terrier": 0,
 "is_popular_dog": 0,
 "is_popular_cat": 1,
 "primary_breed": "Domestic Shorthair",
 "breed_complexity": 1,
 "is_austin_metro": 1,
 "is_core_austin": 0,
 "is_travis_county": 1,
 "is_surrounding_area": 0,
 "is_outside_jurisdiction": 0,
 "distance_category": "Close Suburbs",
 "visit_count": 1,
 "is_return_visit": 0,
 "is_frequent_returner": 0,
 "days_since_last_visit": -1,
 "previous_outcome_type": "First Visit",
 "month_intake": 3,
 "intake_year": 2024,
 "intake_day": 22,
 "intake_dayofweek": 5,
 "birth_year": 2023,
 "birth_month": 9,
 "intake_season": 2,
 "intake_is_weekend": 1,
 "city_area": "Manor",
 "jurisdiction": "TX",
 "date_of_birth": "2023-09-15"
}
```

#### Output Schema
```python
class PredictionResponse(BaseModel):
    adoption_probability: float
    adoption_prediction: str  # "Likely Adopted" or "Unlikely Adopted"
    confidence: str  # "High", "Medium", or "Low"
```

#### Deployment
```bash
# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000

# API Documentation available at
# http://localhost:8000/docs
```

## 🛠 Installation & Setup

### Poetry Dependency Management

This project uses **Poetry** for dependency management, which provides several benefits:

#### What is Poetry?
Poetry is a modern dependency management and packaging tool for Python that:
- **Simplifies dependency resolution** with automatic conflict detection
- **Provides reproducible builds** through lock files
- **Offers virtual environment management** out of the box
- **Streamlines project publishing** and distribution

#### Why Poetry Benefits End Users

1. **Consistent Environment**: Ensures all users have identical dependency versions
2. **Easy Setup**: One command installs all dependencies
3. **Isolation**: Prevents conflicts with system Python packages
4. **Reproducibility**: Lock file guarantees same environment across machines
5. **Modern Standards**: Follows current Python packaging best practices

#### Installation Instructions

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone <repository-url>
cd community-capstone

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the main dashboard
streamlit run main_dashboard.py

# Run the API server
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Alternative Installation (pip)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run applications
streamlit run main_dashboard.py
```

## 📖 Usage Guide

### For Clients

#### Running the Dashboards
1. **Main Dashboard**: `streamlit run main_dashboard.py`
   - Access comprehensive analytics
   - View adoption predictions
   - Explore anomaly detection results

2. **LUCA Dashboard**: `streamlit run streamlit_app_v3.py`
   - Interactive geographic visualization
   - Filter by animal type and outcome
   - Real-time data exploration

3. **Time Series Dashboard**: `streamlit run xgboost_ch_dashboard.py`
   - Forecasting and trend analysis
   - Seasonal pattern identification
   - Performance metrics

#### Using the API
```python
import requests

# Example API call
url = "http://localhost:8000/predict"
data = {
    "animal_type_intake": "Dog",
    "sex_upon_intake": "Intact Male",
    "age_upon_intake_days": 365.0,
    # ... additional features
}

response = requests.post(url, json=data)
result = response.json()
print(f"Adoption Probability: {result['adoption_probability']}")
```

### For Developers

#### Project Structure
```
Community Capstone/
├── main.py                          # FastAPI deployment
├── main_dashboard.py                # Main analytics dashboard
├── streamlit_app_v3.py             # LUCA geographic dashboard
├── time_series_app_v2.py           # Time series analysis
├── pyproject.toml                  # Poetry configuration
├── poetry.lock                     # Dependency lock file
├── Notebooks/                      # Jupyter notebooks
│   ├── Austin ML Model.ipynb       # Austin model development
│   ├── Long Beach EDA.ipynb        # Long Beach analysis
│   ├── Long Beach ML Model.ipynb   # Long Beach model (abandoned)
│   └── Austin_time_series.ipynb    # Time series analysis
└── README.md                       # This file
```

#### Development Workflow
1. **Data Analysis**: Use Jupyter notebooks in `Notebooks/`
2. **Model Development**: Develop in notebooks, export to Python
3. **Dashboard Creation**: Create Streamlit applications
4. **API Development**: Use FastAPI for model deployment
5. **Testing**: Validate models and applications

#### Adding New Features
1. **Data Sources**: Add new data loading functions
2. **Feature Engineering**: Extend feature engineering pipeline
3. **Models**: Add new ML algorithms
4. **Dashboards**: Create new Streamlit applications
5. **API Endpoints**: Extend FastAPI with new endpoints

## 🏗 Technical Architecture

### Technology Stack
- **Backend**: Python 3.11+, FastAPI, H2O.ai
- **Frontend**: Streamlit, Plotly, PyDeck
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: H2O AutoML, XGBoost, Isolation Forest
- **Time Series**: Statsmodels, Prophet
- **Deployment**: Poetry, Uvicorn

### Data Flow
1. **Data Collection**: APIs from various animal shelters
2. **Preprocessing**: Feature engineering and data cleaning
3. **Model Training**: H2O AutoML with cross-validation
4. **Model Deployment**: FastAPI service with H2O model
5. **Visualization**: Streamlit dashboards with real-time data

### Performance Considerations
- **Caching**: Streamlit caching for data loading
- **Optimization**: Efficient data processing pipelines
- **Scalability**: Modular architecture for easy scaling
- **Monitoring**: Health checks and performance metrics

## 🤝 Contributing

### For Future Developers

1. **Code Review**: Follow existing patterns and conventions
2. **Documentation**: Update README and add docstrings
3. **Testing**: Add unit tests for new features
4. **Performance**: Optimize for large datasets
5. **Security**: Follow security best practices

### Development Guidelines
- Use Poetry for dependency management
- Follow PEP 8 coding standards
- Add type hints to functions
- Include comprehensive docstrings
- Test with multiple datasets

### Troubleshooting

#### Common Issues
1. **H2O Model Loading**: Ensure model file path is correct
2. **Data Loading**: Check API endpoints and internet connection
3. **Memory Issues**: Reduce batch sizes for large datasets
4. **Performance**: Use caching and optimize queries

#### Support
- Check the Jupyter notebooks for detailed analysis
- Review API documentation at `/docs` endpoint
- Examine error logs for debugging information

---

**Project Status**: Active Development  
**Last Updated**: July 2025  
**Maintainers**: [@atharvavyas1](https://github.com/atharvavyas1)
