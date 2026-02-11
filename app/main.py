from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os

from app.schemas import CareerRequest, PredictionResponse
from app.service import career_service

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "career_net.pth")
MAPPING_PATH = os.path.join(BASE_DIR, "SAFE_job_mapping.json")

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    try:
        career_service.load_resources(MODEL_PATH, MAPPING_PATH)
        print("System is ready for inference.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(title="Career Inference API", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": career_service.model is not None}

@app.post("/predict/top-matches", response_model=PredictionResponse)
def get_top_matches(payload: CareerRequest):
    """
    Takes 27 personality scores and returns the Top 3 Career Matches.
    """
    if career_service.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # The service now returns the correct Pydantic object directly
        return career_service.predict_top_k(payload.features, k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Note: The above code assumes that the `predict_top_k` method in `CareerModelService` has been updated to return a `PredictionResponse` object directly, which is a more robust and type-safe way to handle API responses.
# it might case an error here I should change something but i dont know what is it 