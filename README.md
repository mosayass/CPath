# Career Inference API

A lightweight, single-endpoint FastAPI microservice for career recommendations powered by a neural network classifier. This service takes personality trait scores as input and returns the top 3 career matches with confidence scores.

## Project Overview

The **Career Inference API** is designed to provide real-time career recommendations based on user personality assessments. It leverages a pre-trained deep neural network model that has been trained on personality-to-career mappings to deliver accurate and personalized career suggestions.

### Key Features

- **Single-endpoint inference**: RESTful API with a dedicated `/predict/top-matches` endpoint
- **Real-time predictions**: Fast inference on personality scores
- **Top-K matching**: Returns the top 3 career matches ranked by confidence scores
- **Health check endpoint**: Monitor service status with the `/` health check endpoint
- **Containerized deployment**: Docker and Docker Compose support for easy deployment
- **Development container**: VS Code Dev Container setup for streamlined development

### Tech Stack

- **Framework**: FastAPI with Uvicorn ASGI server
- **ML Framework**: PyTorch
- **Input validation**: Pydantic models
- **Containerization**: Docker & Docker Compose
- **Development**: VS Code Dev Containers

## Project Structure

```
.
├── app/                          # Main FastAPI application
│   ├── main.py                  # Entry point with lifespan management
│   ├── service.py               # Career model inference service
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── model_arch.py            # Neural network architecture definition
│   └── config.py                # Configuration and settings
│
├── models/                       # Pre-trained model weights
│   ├── career_net (88%).pth     # Trained model checkpoint (88% accuracy)
│   └── .gitkeep                 # Placeholder for Git tracking
│
├── data/                         # Data processing and generation utilities
│   ├── generate_inference_input.py  # Script to create sample inference inputs
│   └── processed/               # Processed data storage
│
├── training/                     # Model training pipeline
│   ├── train.py                 # Training script
│   ├── model_def.py             # Model definition for training
│   ├── dataset.py               # Dataset loading and preprocessing
│   └── __init__.py              # Package initialization
│
├── Dockerfile                    # Docker image build configuration
├── docker-compose.yml            # Docker Compose service orchestration
├── .devcontainer/                # VS Code Dev Container configuration
│   ├── devcontainer.json        # Dev container settings
│   └── docker-compose.yml       # Dev container compose overrides
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── .dockerignore                 # Docker build ignore patterns
```

### Directory Details

#### `/app`
Contains the FastAPI application code responsible for handling HTTP requests and managing model inference.

- **main.py**: Defines FastAPI app with lifespan management for model loading on startup
- **service.py**: Core inference logic with `CareerModelService` singleton
- **schemas.py**: Pydantic models for request validation and response formatting
- **model_arch.py**: PyTorch neural network architecture (`CareerClassifier`)
- **config.py**: Application configuration including model and mapping paths

#### `/models`
Stores pre-trained model weights. The included model achieves 88% accuracy on the career classification task.

#### `/data`
Contains utilities for data processing and inference input generation.

#### `/training`
Includes the complete model training pipeline with dataset handling, model definitions, and training scripts.

## Setup Instructions

### Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 1.29+)
- **VS Code** (for Dev Container setup)

### Quick Start with Docker Compose

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create the Docker network** (required for service communication):
   ```bash
   docker network create ai_shared_network
   ```

3. **Build and start the service**:
   ```bash
   docker-compose up --build
   ```

   The API will be available at `http://localhost:8000`

4. **Verify the service is running**:
   ```bash
   curl http://localhost:8000/
   ```

   Expected response:
   ```json
   {"status": "running", "model_loaded": true}
   ```

### Using Docker Compose (Production)

For production environments, uncomment the `restart: always` policy in `docker-compose.yml`:

```yaml
inference-api:
  # ... other config ...
  restart: always
```

Then run:
```bash
docker-compose up -d
```

### Local Development Setup

#### Option 1: VS Code Dev Container

1. **Install Dev Containers extension** in VS Code
2. **Open the project folder** in VS Code
3. **Reopen in Dev Container**: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
4. **VS Code will**:
   - Build the Docker image
   - Start the container
   - Mount the project folder
   - Initialize the development environment

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

6. **Run the application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   With `--reload` flag enabled, the server will automatically restart when code changes are detected.

#### Option 2: Docker Compose for Development

The `docker-compose.yml` includes volume mounts for hot-reloading:

```yaml
volumes:
  - ./models:/code/models
  - ./app:/code/app
```

This enables live code editing without rebuilding the container.

1. **Start the service**:
   ```bash
   docker-compose up
   ```

2. **Edit files locally** in your editor
3. **Changes reflect immediately** (no rebuild needed)

#### Option 3: Local Python Environment

For minimal setup:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### 1. Health Check Endpoint

**GET** `/`

Returns the service status and model loading state.

**Response**:
```json
{
  "status": "running",
  "model_loaded": true
}
```

### 2. Career Prediction Endpoint

**POST** `/predict/top-matches`

Generates top 3 career recommendations based on personality trait scores.

**Request Body**:
```json
{
  "features": [
    5.2, 4.1, 3.8, 6.0, 2.5, 3.9, 4.2, 5.1, 3.0, 4.5,
    2.8, 5.5, 4.3, 3.7, 5.8, 4.0, 3.5, 5.2, 4.8, 3.2,
    5.0, 4.1, 3.6, 5.3, 2.9, 4.4, 5.1
  ]
}
```

**Constraints**:
- Must provide exactly 27 float values representing personality scores
- Each score must be between 0.0 and 7.0

**Response**:
```json
{
  "top_matches": [
    {
      "rank": 1,
      "job_label": 42,
      "confidence": 0.8234
    },
    {
      "rank": 2,
      "job_label": 157,
      "confidence": 0.0891
    },
    {
      "rank": 3,
      "job_label": 203,
      "confidence": 0.0523
    }
  ]
}
```

**Response Fields**:
- `rank`: Position in the top 3 (1-3)
- `job_label`: Numerical identifier for the recommended career (0-890)
- `confidence`: Probability score (0-1) normalized by softmax

**Error Responses**:
- `400 Bad Request`: Invalid request format or validation failure
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Prediction failure

## Running Tests & Validation

### Generate Sample Inference Input

To create sample input for testing:

```bash
cd data
python generate_inference_input.py
```

### Test the API

Using `curl`:
```bash
curl -X POST http://localhost:8000/predict/top-matches \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      5.2, 4.1, 3.8, 6.0, 2.5, 3.9, 4.2, 5.1, 3.0, 4.5,
      2.8, 5.5, 4.3, 3.7, 5.8, 4.0, 3.5, 5.2, 4.8, 3.2,
      5.0, 4.1, 3.6, 5.3, 2.9, 4.4, 5.1
    ]
  }'
```

Using Python:
```python
import requests

url = "http://localhost:8000/predict/top-matches"
payload = {
    "features": [
        5.2, 4.1, 3.8, 6.0, 2.5, 3.9, 4.2, 5.1, 3.0, 4.5,
        2.8, 5.5, 4.3, 3.7, 5.8, 4.0, 3.5, 5.2, 4.8, 3.2,
        5.0, 4.1, 3.6, 5.3, 2.9, 4.4, 5.1
    ]
}

response = requests.post(url, json=payload)
print(response.json())
```

## Stopping the Service

### Docker Compose

To stop the running containers:
```bash
docker-compose down
```

To stop and remove volumes:
```bash
docker-compose down -v
```

## Configuration

Environment variables can be set in the `docker-compose.yml` file:

```yaml
environment:
  - APP_ENV=development
```

### Model Configuration

- **Input Dimension**: 27 personality trait scores
- **Output Classes**: 891 different careers
- **Model Type**: Multi-layer neural network with batch normalization
- **Accuracy**: 88% on validation set

## Architecture Details

### Model Architecture

The `CareerClassifier` neural network consists of:

1. **Input Layer**: 27 personality traits
2. **Hidden Layer 1**: 512 neurons with batch normalization and ReLU activation
3. **Dropout**: 0.3 dropout rate for regularization
4. **Hidden Layer 2**: 256 neurons with batch normalization and ReLU activation
5. **Output Layer**: 891 neurons (one for each career class)

The model uses softmax activation on outputs to generate probability distributions.

### Inference Flow

1. FastAPI receives HTTP POST request with 27 personality scores
2. Pydantic validates the input schema
3. Input tensor is created from the feature list
4. PyTorch model processes the input with batch inference
5. Softmax probabilities are computed
6. Top-3 matches are extracted with their confidence scores
7. Results are formatted and returned as JSON response

## Network Configuration

The service uses a shared Docker network (`ai_shared_network`) to communicate with other services (e.g., a C# backend). The network is marked as `external` in `docker-compose.yml`, meaning it must be created separately:

```bash
docker network create ai_shared_network
```

## Troubleshooting

### Model Not Loading

If you see `CRITICAL ERROR: Failed to load model`:

1. Verify the model file exists: `models/career_net (88%).pth`
2. Check file permissions: `ls -la models/`
3. Ensure sufficient disk space
4. Review console output for specific error messages

### Connection Issues

If the API is unreachable:

1. **Check if containers are running**:
   ```bash
   docker ps
   ```

2. **Verify network exists**:
   ```bash
   docker network ls | grep ai_shared_network
   ```

3. **Check logs**:
   ```bash
   docker-compose logs inference-api
   ```

### Prediction Errors (503 Service Unavailable)

The model may not have loaded successfully:

1. Check the health endpoint: `curl http://localhost:8000/`
2. Look for error messages in container logs
3. Verify `MAPPING_PATH` exists if job mapping is required

### Invalid Request Format

Ensure your request has exactly 27 features between 0.0 and 7.0:

```bash
# Count features
python -c "import json; data={'features': [...]}; print(len(data['features']))"
```

## Performance Notes

- **Inference Time**: ~10-50ms per prediction (CPU)
- **Memory Usage**: ~500MB for model + runtime
- **Concurrency**: FastAPI/Uvicorn handles multiple concurrent requests efficiently
- **Scalability**: Can be scaled horizontally with load balancing

## Future Enhancements

- Batch prediction endpoint for multiple inputs
- Model versioning and A/B testing support
- Request/response logging and monitoring
- Caching layer for frequently requested predictions
- GPU support for faster inference
- Model fine-tuning endpoints

## License


## Contributing


## Support

For issues, questions, or contributions, please [contact/open an issue].
