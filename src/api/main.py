from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
label_encoders = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    global model, label_encoders, feature_names
    
    logger.info("Starting up - Loading model...")
    try:
        logger.info("Loading model from MLflow Registry...")
        model_uri = "models:/attrition_model/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded from MLflow Registry")
        
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        logger.info("✅ Encoders and features loaded")
        
    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        logger.info("Trying to load from local file...")
        
        try:
            model = joblib.load('models/model.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            logger.info("✅ Model loaded from local files")
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
    
    yield
    logger.info("Shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Attrition Prediction API",
    description="Predict employee attrition using ML",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files (for serving web UI)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse('src/static/index.html')

# Keep all your existing endpoints below...
# (Employee model, /health, /predict, etc.)

# Pydantic model for request
class Employee(BaseModel):
    age: int
    business_travel: str
    department: str
    distance_from_home: int
    education: int
    education_field: str
    job_role: str
    marital_status: str
    monthly_income: int
    over_time: str
    years_at_company: int
    # Add other features your model uses
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "business_travel": "Travel_Rarely",
                "department": "Sales",
                "distance_from_home": 10,
                "education": 3,
                "education_field": "Life Sciences",
                "job_role": "Sales Executive",
                "marital_status": "Married",
                "monthly_income": 5000,
                "over_time": "No",
                "years_at_company": 5
            }
        }

# Response model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str

@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Attrition Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: Employee):
    """Predict attrition for a single employee"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        data = employee.dict()
        df = pd.DataFrame([data])
        
        # Apply label encoding for categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure all features are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability (if available)
        try:
            probability = float(model.predict_proba(df)[0][1])
        except:
            # If model doesn't have predict_proba (e.g., MLflow wrapper)
            probability = float(prediction)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction="Will Leave" if prediction == 1 else "Will Stay",
            probability=round(probability, 4),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(employees: List[Employee]):
    """Predict attrition for multiple employees"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for emp in employees:
        try:
            result = await predict(emp)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    
    return {
        "total": len(results),
        "predictions": results
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "features_count": len(feature_names) if feature_names else 0,
        "features": feature_names[:10] if feature_names else [],  # First 10
        "categorical_features": list(label_encoders.keys()) if label_encoders else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)