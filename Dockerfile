# Prophet Forecasting Agent - Dockerfile
# Python 3.11 slim base for minimal image size and fast startup

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (none needed for this project, but keep layer for future use)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Expose the default port (configurable via PROPHET_PORT env var)
EXPOSE 8080

# Run the FastAPI application with uvicorn
# Host 0.0.0.0 ensures the container accepts external connections
# Port is configurable via PROPHET_PORT (default 8080)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
