# Import necessary libraries
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from rameshm.llmeng.utils.init_utils import set_environment_logger

class LlmChat:
    def __init__(self, chat_id: int, model_nm: str, history: Optional[List[Dict[str, Any]]] = None, system_message: str = ""):
        #  Initialize the logger and sets environment variables
        self.logger = set_environment_logger()
        self.chat_id = chat_id
        self.history = history.copy() if history is not None else []
        self.model_nm = model_nm
        self.system_message = system_message
        self.title = self.create_chat_title()
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def get_model(self) -> str:
        return self.model_nm
    def get_chat_id(self) -> int:
        return self.chat_id
    def get_model_nm(self) -> str:
        return self.model_nm
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    def get_chat_title(self) -> str:
        return self.title

    def create_chat_title(self, max_length: int = 50) -> str:
        """Generate chat title from first user message"""
        if not self.history:
            return f"New Chat - {self.get_model_nm()}_{datetime.now().strftime('%H:%M')}"

        first_user_msg = next((msg['content'] for msg in self.history if msg['role'] == 'user'), "")
        if first_user_msg:
            title = self.get_model_nm()
            title += "-" + first_user_msg[:max_length]
            if len(first_user_msg) > max_length - len(self.get_model_nm()):
                title += "..."
            return title
        return f"Chat - {datetime.now().strftime('%H:%M')}"

    def update_chat_history(self, history: List[Dict[str, Any]]):
        """Update chat history"""
        self.history = history.copy()
        self.updated_at = datetime.now().isoformat(timespec="seconds")
