import torch
import json
import os
import torch.nn.functional as F
from app.model_arch import CareerClassifier
# NEW: Import the schemas
from app.schemas import CareerPrediction, PredictionResponse 

class CareerModelService:
    def __init__(self):
        self.model = None
        self.mapping = None
        self.idx_to_job = {}

    def load_resources(self, model_path: str, mapping_path: str):
        # ... (This part remains exactly the same as before) ...
        print(f"Loading resources from: {model_path}")
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")
            
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
            
        self.idx_to_job = {v: k for k, v in self.mapping.items()}
        num_classes = len(self.mapping)

        self.model = CareerClassifier(input_dim=28, num_classes=num_classes)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print("Model loaded successfully.")

    # UPDATED METHOD
    def predict_top_k(self, features: list, k=3) -> PredictionResponse:
        """
        Infers the top K career matches and returns a strict Pydantic Response.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded!")

        # Convert list to Tensor
        input_tensor = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k)

        # Build list of CareerPrediction objects
        matches = []
        for i in range(k):
            idx = top_indices[0][i].item()
            score = top_probs[0][i].item()
            job_title = self.idx_to_job.get(idx, "Unknown")
            
            # Create strict Pydantic object
            prediction = CareerPrediction(
                rank=i + 1,
                job_label=idx,
                job_title=job_title,
                confidence=round(score, 4)
            )
            matches.append(prediction)
            
        # Return the wrapper response
        return PredictionResponse(top_matches=matches)

# Singleton Instance
career_service = CareerModelService()