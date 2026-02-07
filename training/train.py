import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import os
from dataset import CareerDataset

# --- CONFIGURATION ---
# Paths are relative to the 'training/' folder
CSV_PATH = "../SAFE_training_data_50k.csv"
MAP_PATH = "../SAFE_job_mapping.json"
MODEL_SAVE_DIR = "../models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "career_net.pth")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
INPUT_DIM = 27  # 6 RIASEC + 22 Work Styles

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
        
        # Layer 1: Expand significantly to capture trait combinations
        # 28 -> 512
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512) # Optional: Helps training stability
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Increased slightly for wider layer
        
        # Layer 2: Keep it wide to maintain class separation
        # 512 -> 256
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Output Layer
        # 256 -> ~900
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # Layer 1
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Layer 2
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Output
        out = self.output_layer(out)
        return out

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            
            scores = model(features)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum().item()
            num_samples += predictions.size(0)
    
    model.train()  # Return to training mode
    return float(num_correct) / float(num_samples) * 100 
# --- TRAINING LOOP ---
def train():
    print("--- Starting Phase 2: Training ---")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Prepare Data
    print("Loading Dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Data file not found at {CSV_PATH}. Make sure you are running this from the 'training/' folder.")

    dataset = CareerDataset(CSV_PATH)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # --- NEW: Create Two Loaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Don't shuffle test data
    
    print(f"Data Loaded: {len(dataset)} total.")
    print(f"Training on {len(dataset)} samples | Testing on {len(test_dataset)} samples.")

    

    # 2. Initialize Model
    num_classes = get_num_classes(MAP_PATH)
    print(f"Detected {num_classes} Job Classes.")
    
    model = CareerClassifier(INPUT_DIM, num_classes).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # ---  CHECK REAL ACCURACY ---
    print("--- Evaluating on Test Set (Unseen Data) ---")
    acc = check_accuracy(test_loader, model, device)
    print(f"FINAL ACCURACY: {acc:.2f}%")
    # --- 4. SAVE MODEL ---
    print("--- Saving Model ---")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved successfully to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
