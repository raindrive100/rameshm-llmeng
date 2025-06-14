# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:45:11 2025
@author: rameshUser

Loads environment variables for the llm_engineering project.
"""
# imports external and system packages
import os

# import internal packages
from dotenv import load_dotenv

def set_my_environment():
    # Load the file containing Keys for multiple environments needed
    key_file = os.getenv("LLM_KEY_FILE")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Key file not found: {key_file}")
    load_dotenv(key_file)

    os.environ.update({
        "MAX_SIZE_OF_FILE_UPLOADS": str(1024*1024*20),   # 20MB max total size of all uploaded files.
        "LLM_MY_LOGGER_NAME": "llm_engineering",
        "LLM_LOG_TO_FILE": "True",
        "LLM_LOG_DIR": "c:\\temp",
        "LLM_LOG_FILE_NM": "my_logs.txt",
        "LLM_LOG_FILE_MAX_BITE_SIZE": str(1024*1024*10), # 10MB Log File size then it rotates
        "LLM_LOG_FILE_BKUP_CNT": str(5),
        "LLM_APP_LOG_LEVEL": "DEBUG",
        "LLM_EXTERNAL_LIB_LOG_LEVEL": "WARNING" #Log level for packages and libs developed by others
    })

if __name__ == "__main__":
    set_my_environment()