from typing import List

from pydantic import BaseModel, Field, field_validator


# 1. Input Schema
class CareerRequest(BaseModel):
    # We expect 27 float values (Indices 0-26)
    features: List[float] = Field(
        ..., 
        min_length=27, 
        max_length=27,
        description="Array of 27 personality scores "
    )

    @field_validator('features')
    def validate_range(cls, v):
        # Optional: Ensure values are within 0-7 scale if that's your logic
        if any(x < 0.0 or x > 7.0 for x in v):
            raise ValueError("All scores must be between 0.0 and 7.0")
        return v

# 2. Output Schema (Nested)
class CareerPrediction(BaseModel):
    rank: int
    job_label: int
    confidence: float

class PredictionResponse(BaseModel):
    top_matches: List[CareerPrediction]