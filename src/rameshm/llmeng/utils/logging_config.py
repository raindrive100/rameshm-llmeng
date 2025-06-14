# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:28:38 2025

@author: rameshUser

Log Setting to be used by all the modules

"""

#imports
import logging
import os
from rameshm.llmeng.utils import set_environment


class MyLogger:
    def __init__(self, log_file_nm: str = None): 
        """
        Initialize the logger with a default or provided log file name.
        """
        self.log_file_nm = log_file_nm or os.path.join(
            os.getenv("LLM_LOG_DIR"), os.getenv("LLM_LOG_FILE_NM")
            )
        print(f"Log File: {self.log_file_nm}")
        
    def get_logger(self, name: str) -> logging.Logger: # We need to initialize the logger with made it required
        """
        Configure and return a logger with the specified name.
        """
        log_config = self.__get_log_config()
        logging.config.dictConfig(log_config)
        return logging.getLogger(name)
            
    def __get_log_config(self) -> dict:
        """
        Define and return Log Config
        """
        log_to_file = os.getenv("LLM_LOG_TO_FILE", "false").lower() == "true"
        log_handlers = ["fileHandler"] if log_to_file else ["consoleStdOut"]
           
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
              "level": f'{os.getenv("LLM_APP_LOG_LEVEL")}',
              "stream": "ext://sys.stdout"
            },
            "consoleStdErr": {
              "class": "logging.StreamHandler",
              "formatter": "standard",
              "level": f'{os.getenv("LLM_APP_LOG_LEVEL")}',
              "stream": "ext://sys.stderr"
            },
            "fileHandler": {
              "class": "logging.handlers.RotatingFileHandler",
              "formatter": "standard",
              "level": f'{os.getenv("LLM_APP_LOG_LEVEL")}',
              "filename": f"{self.log_file_nm}",
              "encoding": "utf8",
              "maxBytes": int(os.getenv("LLM_LOG_FILE_MAX_BITE_SIZE", 1024*1024*10)), #10MB Log file
              "backupCount": int(os.getenv("LLM_LOG_FILE_BKUP_CNT", 5))
            }
          },
          "loggers": {
              "llm_engineering": {  # Specifi logger for llm_engineering project
                "handlers": log_handlers,
                "level": f'{os.getenv("LLM_APP_LOG_LEVEL")}',
                "propagate": False  # Don't propagate to root
                }
              },
          "root": {
            "handlers": log_handlers,
            "level": f'{os.getenv("LLM_EXTERNAL_LIB_LOG_LEVEL")}'
          }
        }
        return log_config
        
if __name__ == "__main__":
    set_environment.set_my_environment()
    logger = MyLogger().get_logger(os.getenv("LLM_MY_LOGGER_NAME"))
    logger.debug("Testing Debug Log")
    logger.info("Testing Info Log")
    logger.warning("Testing Warning Log")
