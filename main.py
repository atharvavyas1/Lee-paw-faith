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
model = h2o.load_model("StackedEnsemble_AllModels_1_AutoML_1_20250720_170401")

# Define input schema
class AnimalFeatures(BaseModel):
    animal_type_intake: str
    sex_upon_intake: str
    age_upon_intake_days: float
    intake_type: str
    intake_condition: str
    has_name: int
    is_mixed: int
    size_category: str
    is_working: int
    is_sporting: int
    is_terrier: int
    is_popular_dog: int
    is_popular_cat: int
    primary_breed: str
    is_austin_metro: int
    is_core_austin: int
    distance_category: str
    visit_count: int
    is_return_visit: int
    days_since_last_visit: float
    previous_outcome_type: str
    month_intake: int
    intake_year: int
    intake_day: int
    birth_year: int
    birth_month: int
    intake_season: int
    intake_is_weekend: int
    city_area: str
    jurisdiction: str

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
        
        # Set categorical columns
        categorical_cols = ['animal_type_intake', 'sex_upon_intake', 'intake_type', 
                           'intake_condition', 'size_category', 'primary_breed',
                           'distance_category', 'previous_outcome_type', 'city_area', 'jurisdiction']
        
        for col in categorical_cols:
            if col in h2o_frame.columns:
                h2o_frame[col] = h2o_frame[col].asfactor()
        
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
