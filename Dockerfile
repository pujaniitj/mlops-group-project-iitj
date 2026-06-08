#Docker file code:


# Use Python 3.11 slim as base image (lightweight)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Build-time argument: which HF model to load
# Can be overridden at build time with --build-arg
ARG HF_MODEL_NAME=pujaniitj/mlops-group-sentiment

# Make the argument available at runtime as environment variable
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Install Python packages
# Using CPU-only torch to keep the image size under 2GB
RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    huggingface_hub>=0.24.0

# Copy the source scripts into the container
COPY src/ ./src/

# Default command: run the inference script
CMD ["python", "src/inference.py"]
