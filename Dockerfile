# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       pkg-config \
       python3-dev \
       libssl-dev \
       libffi-dev \
       libjpeg-dev \
       zlib1g-dev \
       libopenblas-dev \
       liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements first to allow Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=120 --retries=10 -r requirements.txt

# Now copy the rest of the code
# COPY . /app


# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]