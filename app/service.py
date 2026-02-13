import torch
import os
import torch.nn.functional as F
from app.model_arch import CareerClassifier
from app.schemas import CareerPrediction, PredictionResponse 

class CareerModelService:
    def __init__(self):
        self.model = None
        self.mapping = None

    def load_resources(self, model_path: str, mapping_path: str):
        # ... (This part remains exactly the same as before) ...
        print(f"Loading resources from: {model_path}")

        self.model = CareerClassifier(input_dim=28, num_classes=891)
        
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
            
            # Create strict Pydantic object
            prediction = CareerPrediction(
                rank=i + 1,
                job_label=idx,
                confidence=round(score, 4)
            )
            matches.append(prediction)
            
        # Return the wrapper response
        return PredictionResponse(top_matches=matches)

# Singleton Instance
career_service = CareerModelService()