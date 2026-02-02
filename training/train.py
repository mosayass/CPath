import torch
import torch.nn as nn
import json
import os

# 1. Helper to determine output size dynamically
def get_num_classes(mapping_path):
    """
    Reads the JSON mapping file to count how many distinct job labels exist.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")
        
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    # The length of the dictionary keys/values tells us the number of classes
    return len(mapping)

# 2. The Neural Network Architecture
class CareerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CareerClassifier, self).__init__()
        
        # Input Layer (28 Features) -> Hidden Layer 1 (128 Neurons)
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        
        # Hidden Layer 1 -> Hidden Layer 2 (64 Neurons)
        # We taper down the size to force the model to learn efficiently
        self.layer2 = nn.Linear(128, 64)
        
        # Hidden Layer 2 -> Output Layer (num_classes)
        # No Softmax here because CrossEntropyLoss includes it automatically
        self.output_layer = nn.Linear(64, num_classes)
        
        # Optional: Add Dropout if overfitting occurs later
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        # Pass through Layer 1
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Pass through Layer 2
        out = self.layer2(out)
        out = self.relu(out)
        
        # Pass through Output Layer
        out = self.output_layer(out)
        return out
