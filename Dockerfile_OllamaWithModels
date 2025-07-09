# I have adopted a multi-build process to see if the image size can be reuced from 7GB to a smaller size, but no luck.
# To keep to simple we can just include the first two commands and it will still work.
# TODO: Try a Slim Ollama image and see if that works or just a CPU version. For now keeping the multi-build appraoch.
#FROM ollama/ollama AS model_builder
FROM ollama/ollama

# Step 1: Start the Ollama server in the background
# Step 2: Wait briefly for it to initialize (optional, but good practice for robustness)
# Step 3: Run the ollama pull commands
# Step 4: (Implicitly, the server process will be stopped when the RUN command finishes)
RUN (ollama serve &) && \
    # Give Ollama a moment (10sec) to start up (adjust sleep if needed)
    sleep 10 && \
    ollama pull llama3.2:1b && \
    ollama pull gemma3:1b && \
    kill $(pgrep ollama) || true # Gracefully stop Ollama server
    #ollama pull ai/llama3.2 &&  \
    #ollama pull ai/gemma3:1b

# Update package lists and install vim
RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# THE ENTIRE BLOCK FROM HERE ON DOWN CAN BE COMMENTED OUT AND IT WILL STILL WORK. But left it as an example of
# MULTI_STAGE BUILD.
# --- Runtime Stage ---
# This is the NEW and CRUCIAL part: A second FROM instruction for the final image.
# Start a new stage, which will be your final image.
#FROM ollama/ollama
#
## Copy the entire .ollama directory contents from the build stage
## This will include 'models', 'blobs' (if nested), 'manifests', 'history', and 'id_ed25519' files.
#COPY --from=model_builder /root/.ollama/ /root/.ollama/
#
## Set the OLLAMA_MODELS environment variable to point to the correct directory
#ENV OLLAMA_MODELS=/root/.ollama
#
## The Ollama will continue to work even if we exclude the below Expose, Entrypoint, and CMD sections because
## they are already part of ollama/ollma image. Included here for clarity and readability.
#
## Expose the port (already done by base image, but good to be explicit)
#EXPOSE 11434
#
## Command to run Ollama when the container starts
#ENTRYPOINT ["/bin/ollama"]
#CMD ["serve"]
#

# Add more RUN ollama pull commands for other models