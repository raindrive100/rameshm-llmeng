# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:28:38 2025

@author: rameshUser

Log Setting to be used by all the modules

"""

# imports
import logging
from logging import config as logging_config
import os


class MyLogger:
    def __init__(self, log_file_nm: str = None, log_level: str = "INFO", log_to_file_only: bool = False,
                 external_lib_log_level: str = "WARNING"):
        """
        log_file_nm: Optional full path to the log file. If provided, logs will be written to this file.
        log_level: Optional log level (default is "INFO"). Can be set to "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        log_to_file_only: If True, logs will only be written to the file specified by log_file_nm else to console.
        external_lib_log_level: Set it the desired level.
        """
        self.log_file_nm = log_file_nm
        self.log_level = log_level.upper()
        self.log_to_file_only = log_to_file_only
        self.external_lib_log_level = external_lib_log_level


    def get_logger(self, name: str) -> logging.Logger:  # We need to initialize the logger with made it required
        """
        Configure and return a logger with the specified name.
        """
        log_config = self.__get_log_config()
        logging_config.dictConfig(log_config)
        return logging.getLogger(name)


    def __get_log_config(self) -> dict:
        """
        Define and return Log Config
        """
        DEFAULT_LOG_LEVEL = "INFO"
        log_to_file = self.log_file_nm == "true"
        log_to_console = self.log_to_file_only == False

        if self.log_to_file_only and not self.log_file_nm:
            raise ValueError("Log file name must be provided if log_to_file_only is True.")
        # Determine log handlers based on log_file_nm and log_to_file_only
        if self.log_file_nm and self.log_to_file_only == False:
            log_handlers = ["fileHandler", "consoleStdOut", "consoleStdErr"]
        elif self.log_file_nm and self.log_to_file_only:
            log_handlers = ["fileHandler"]
        else:
            log_handlers = ["consoleStdOut", "consoleStdErr"]

        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(process)d - %(name)s - %(filename)s - %(funcName)s - %(lineno)d"
                              "- %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "consoleStdOut": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": f'{self.log_level}',   # Messages at Warning and above will be logged to stdout and stderr so we will see them twice in console
                    "stream": "ext://sys.stdout"
                },
                "consoleStdErr": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": f'WARNING',  # Default level for stderr
                    "stream": "ext://sys.stderr"
                },
                "fileHandler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "level": f'{self.log_level}',
                    "filename": f"{self.log_file_nm}",
                    "encoding": "utf8",
                    "maxBytes": int(os.getenv("LLM_LOG_FILE_MAX_BITE_SIZE", 1024 * 1024 * 10)),  # 10MB Log file
                    "backupCount": int(os.getenv("LLM_LOG_FILE_BKUP_CNT", 5))
                },
                # Instead of specifying log handler for each external library, I am using a fileHandler instead.
                "fileHandlerExternalLibs": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "level": f'{self.log_level}',
                    "filename": f"{self.external_lib_log_level}",
                    "encoding": "utf8",
                    "maxBytes": int(os.getenv("LLM_LOG_FILE_MAX_BITE_SIZE", 1024 * 1024 * 10)),  # 10MB Log file
                    "backupCount": int(os.getenv("LLM_LOG_FILE_BKUP_CNT", 5))
                }
            },
            "loggers": {
                "llm_engineering": {  # Specific logger for llm_engineering project
                    "handlers": log_handlers,
                    "level": f'{self.log_level}',
                    "propagate": False  # Don't propagate to root
                }
            },
            "root": {
                "handlers": ["fileHandlerExternalLibs"],
                "level": f'{self.external_lib_log_level}'
            }
        }
        return log_config


if __name__ == "__main__":
    logger = MyLogger(log_level="DEBUG").get_logger("llm_engineering")
    logger.debug("Testing Debug Log")
    logger.info("Testing Info Log")
    logger.warning("Testing Warning Log")
