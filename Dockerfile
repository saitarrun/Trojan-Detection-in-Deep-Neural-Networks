# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some scientific libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies, adding FastAPI and scikit-learn
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn python-multipart scikit-learn

# Copy the current directory contents into the container
COPY . .

# Expose port (can be overridden by docker-compose)
EXPOSE 8000
EXPOSE 8501

# By default, run the FastAPI backend, but allow docker-compose to override
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
