# Dockerfile for Walker2d TD3/EAS-TD3 Training
# Usage: docker build -t walker2d-td3 .
#        docker run --gpus all -it walker2d-td3

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy project source code
COPY src/ ./src/

# Create output directories
RUN mkdir -p models results videos

# Run training by default
CMD ["python", "-m", "src.run_experiment", "--algorithm", "td3", "--timesteps", "500000"]
