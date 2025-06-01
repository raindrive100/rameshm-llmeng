# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:42:41 2025

@author: rameshUser

Initializes environment variables and creates a shared logger.
"""
# imports external and system packages
import os

# import internal packages
from rameshm.llmeng.utils.set_environment import set_my_environment
from rameshm.llmeng.utils.logging_config import MyLogger

def get_initialized_logger():
    """Initialize environment and return a configured logger."""
    set_my_environment()
    return MyLogger().get_logger(os.getenv("LLM_MY_LOGGER_NAME"))

if __name__ == "__main__":
    logger = get_initialized_logger()
    logger.warning(f"Testing Logger with WARNING level. Logger is: {logger.name}")