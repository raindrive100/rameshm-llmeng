# syntax=docker/dockerfile:1.2 # Add this at the top if using BuildKit cache mounts
FROM python:3.11.12-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements_llmchat.txt requirements.txt

# Install Python packages from requirements.txt
# Using cache mount with BuildKit (optional but recommended)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --target=/app/libs -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update package lists and install vim
RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# Add /app to PYTHONPATH so Python can find your custom packages
ENV PYTHONPATH="${PYTHONPATH}:/app/libs"

# Add metadata labels (optional)
LABEL description="Bsse Python image with packages needed for my LLM"