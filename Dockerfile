
# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout (better for logging)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Note: Because of .dockerignore, data/ and training/ are NOT copied here
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
