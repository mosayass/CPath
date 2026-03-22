import json

# Absolute path to the file
file_path = r'C:\Users\mosa\OneDrive\Desktop\Graduation Project\TrainingData\SAFE_job_mapping.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Counting the top-level keys
    print(len(data))
    
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
except json.JSONDecodeError:
    print("Error: The file is not a valid JSON.")