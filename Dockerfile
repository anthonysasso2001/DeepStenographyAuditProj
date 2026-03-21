# Use a versioned base image for stability (instead of :latest)
FROM tensorflow/tensorflow:latest-gpu-jupyter


# 2. Install python and system dependencies
# We use --no-install-recommends to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libcudnn8 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
COPY . /app

# 3. Install python packages
# Upgrade pip first, then install from requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt