# syntax=docker/dockerfile:1.2 # Add this at the top if using BuildKit cache mounts
# Dockerfile for LLM Chat Application
# Use a lightweight Python image as the base
# Important: The Context for this Dockerfile should be set to the parent directory of `src` and `requirements_llmchat.txt`.
# This allows the Dockerfile to access the `src` directory and the requirements file. See the llmchat.compose.yml for context setting.

FROM python:3.11.12-slim-bullseye

# Set a build argument for the environment type, defaulting to PROD.
ARG BUILD_ENV=PROD

# Set the working directory
WORKDIR /app

# Copy the requirements file into the working directory
COPY docker-files/requirements_llmchat.txt requirements.txt

# Install Python packages from requirements.txt
# Using cache mount with BuildKit (optional but recommended)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --target=/app/libs -r /app/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update package lists and install vim
RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY src/ /app/src/

# Add /app to PYTHONPATH so Python can find your custom packages
ENV PYTHONPATH="${PYTHONPATH}:/app/src:/app/libs"

#COPY .env /app/secrets.env \

# Copy the entrypoint script
COPY docker-files/entrypoint_llmchat.sh entrypoint_llmchat.sh

# Expose the port your Gradio app uses (usually 7860)
EXPOSE 7860

# Set entrypoint (if used with ENTRYPOINT)
ENTRYPOINT ["/app/entrypoint_llmchat.sh"]

# Set the command to run the Gradio app (using exec form)
CMD ["python", "/app/src/rameshm/llmeng/llmchat/gr_ui.py"]

# Add metadata labels (optional)
LABEL description="LLM Chat application with Gradio"