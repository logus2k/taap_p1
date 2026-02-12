# STAGE 1: Builder
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Use the native Python version for Ubuntu 24.04 (Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake git gcc-12 g++-12 \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv using the native python3
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir matplotlib "torchmetrics[image]" pydot python-socketio fastapi uvicorn

# STAGE 2: Runtime
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 \
    gnupg curl \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment
COPY --from=builder /opt/venv /opt/venv

# Set Environment Variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda

WORKDIR /
