## Use the official Python slim image as the base
#FROM python:3.11.12-slim-bullseye
#
## Install necessary tools and Micromamba
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    curl \
#    ca-certificates && \
#    rm -rf /var/lib/apt/lists/* && \
#    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /bin --strip-components=1 bin/micromamba
#
#RUN micromamba --version
#
## Set the environment variable for Micromamba's environment directory
#ENV MAMBA_ROOT_PREFIX=/opt/conda

# Use miniconda as base image
#FROM continuumio/miniconda3:latest
FROM mambaorg/micromamba:latest AS builder
#
RUN micromamba --version

# Create the environment and install dependencies
COPY environment_llmchat.yml /tmp/environment.yml
RUN micromamba create -v --yes --file /tmp/environment.yml && \
    micromamba clean --all --yes

# Set the working directory
WORKDIR /app

# Copy the application code
COPY src/ .

# Copy secrets file (if necessary)
COPY .env /app/secrets.env

# Copy the entrypoint script
COPY entrypoint_llmchat.sh /entrypoint_llmchat.sh

# Make the entrypoint script executable
#RUN chmod +x /app/entrypoint_llmchat.sh

# Expose the port your Gradio app uses (usually 7860)
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["/entrypoint_llm.sh"]
CMD ["python", "src/rameshm/llmengineering/llmchat/gr_ui.py"]