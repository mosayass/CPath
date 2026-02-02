import torch
from torch.utils.data import Dataset
import pandas as pd

class CareerDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
        """
        # Load the clean data
        self.data = pd.read_csv(csv_path)

        # Separate Features and Target
        # Features: All columns except 'Job_Label' (The 28 RIASEC/Work Style columns)
        # We assume the CSV is perfectly clean based on Phase 1 verification
        self.X = self.data.drop(columns=['Job_Label']).values
        
        # Target: The 'Job_Label' column (Integers 0-890)
        self.y = self.data['Job_Label'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert row 'idx' to tensors
        # Features must be Float (for the Neural Network weights)
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        
        # Labels must be Long (integers) for CrossEntropyLoss classification
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return features, label
