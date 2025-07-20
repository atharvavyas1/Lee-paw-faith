from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import h2o
import pandas as pd
from typing import Optional

# Initialize FastAPI
app = FastAPI(title="Animal Adoption Prediction API")

# Initialize H2O (do this once at startup)
h2o.init()

# Load your saved model
model = h2o.load_model("/Users/atharvavyas/Desktop/McGill Academics/Community Capstone/StackedEnsemble_AllModels_1_AutoML_1_20250720_170401")

# Define input schema
class AnimalFeatures(BaseModel):
    animal_type_intake: str
    sex_upon_intake: str
    age_upon_intake_days: float
    intake_type: str
    intake_condition: str
    has_name: int
    is_mixed: int
    num_breeds: int
    size_category: str
    is_working: int
    is_sporting: int
    is_terrier: int
    is_popular_dog: int
    is_popular_cat: int
    primary_breed: str
    breed_complexity: int
    is_austin_metro: int
    is_core_austin: int
    is_travis_county: int
    is_surrounding_area: int
    is_outside_jurisdiction: int
    distance_category: str
    visit_count: int
    is_return_visit: int
    is_frequent_returner: int
    days_since_last_visit: float
    previous_outcome_type: str
    month_intake: int
    intake_year: int
    intake_day: int
    intake_dayofweek: int
    birth_year: int
    birth_month: int
    intake_season: int
    intake_is_weekend: int
    city_area: str
    jurisdiction: str
    date_of_birth: str  # Add as string, convert to datetime

# Define response schema
class PredictionResponse(BaseModel):
    adoption_probability: float
    adoption_prediction: str
    confidence: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_adoption(features: AnimalFeatures):
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Convert to H2O Frame
        h2o_frame = h2o.H2OFrame(input_data)
        
        # Set categorical columns (only string/text columns)
        # Set categorical columns (only object types from training)
        categorical_cols = [
            'animal_type_intake', 'sex_upon_intake', 'intake_type', 
            'intake_condition', 'size_category', 'primary_breed',
            'distance_category', 'previous_outcome_type', 
            'city_area', 'jurisdiction', 'is_mixed', 'is_working', 'is_sporting', 'is_terrier', 'is_popular_dog', 'is_popular_cat', 
            'is_austin_metro', 'is_core_austin', 'is_travis_county', 'is_surrounding_area', 'is_outside_jurisdiction'
        ]



        
        for col in categorical_cols:
            if col in h2o_frame.columns:
                h2o_frame[col] = h2o_frame[col].asfactor()
        
        # Make prediction
        # Optional: Validate model compatibility
        # Get feature names only (exclude target)
        # Safely get target column
        output_meta = model._model_json['output']
        target_column = output_meta['response_column_name'] if 'response_column_name' in output_meta else model.actual_params['response_column']

        # Get model input columns excluding target
        model_columns = output_meta['names']
        model_feature_columns = [col for col in model_columns if col != target_column]
        print("Expected columns:", model_feature_columns)
        print("Provided columns:", h2o_frame.columns)
        # Check if any expected columns are missing
        input_columns = h2o_frame.columns
        missing_cols = set(model_feature_columns) - set(input_columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns in input: {missing_cols}")

        # Reorder input columns to match training order
        h2o_frame = h2o_frame[model_feature_columns]


        # Make prediction
        prediction = model.predict(h2o_frame)
        prob = prediction.as_data_frame()['predict'][0]
        
        # Determine prediction and confidence
        adoption_pred = "Likely Adopted" if prob > 0.5 else "Unlikely Adopted"
        confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
        
        return PredictionResponse(
            adoption_probability=round(prob, 4),
            adoption_prediction=adoption_pred,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Animal Adoption Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}