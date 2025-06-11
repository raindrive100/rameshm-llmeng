#from rameshm.llmeng.utils.init_utils import set_environment_logger

class LlmChatException(Exception):
    """Base class for all exceptions in the LLM chat module."""
    def __init__(self, message: str):
        super().__init__(message)
        #self.logger = set_environment_logger()
        #self.logger.error(message)