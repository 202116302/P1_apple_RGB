FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web/ ./web/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create uploads directory
RUN mkdir -p /app/web/uploads

# Set working directory to web
WORKDIR /app/web

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
