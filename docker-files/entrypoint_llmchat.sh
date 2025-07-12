#!/bin/bash

set -e

# Activate the conda environment
#micromamba activate llms

export LLM_KEY_FILE="/app/secrets.env"
export RUN_ENVIRONMENT="PROD"
## Load environment variables from file if it exists
#if [ -f /app/secrets.env ]; then
#    echo "Loading environment variables from secrets.env..."
#    source /app/secrets.env
#fi

# Execute the original command
exec "$@"