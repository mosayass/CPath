import os

class Settings:
    # Paths are relative to the root where docker/uvicorn runs
    # We assume the app is run from the project root (CPath/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    MODEL_PATH = os.path.join(BASE_DIR, "models", "career_net (88%).pth")
    
    # Model Parameters
    INPUT_DIM = 27
    
settings = Settings()