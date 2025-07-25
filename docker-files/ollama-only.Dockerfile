# This image contains the full version of Ollama which is deployed into Kubernetes as ollama-only-full.
# The image generated from this is stored in Artifact as ollma-only-full and is used in ollama-with-models-deployment.yaml file.
#FROM alpine/ollama
FROM ollama/ollama

# Update package lists and install vim
RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*


# Expose the default Ollama port
#EXPOSE 11434

# Command to run the Ollama server
#CMD ["ollama", "serve"]