# Import necessary libraries
from langchain_community.chat_models.friendli import get_chat_request
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os
from dotenv import load_dotenv
import time
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from rameshm.llmeng.utils.init_utils import set_environment_logger

class LlmChat:
    def __init__(self, chat_id: str, model_nm: str, history: Optional[List[Dict[str, Any]]] = None, system_message: str = ""):
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
    def get_chat_id(self) -> str:
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
            return f"New Chat - {datetime.now().strftime('%H:%M')}"

        first_user_msg = next((msg['content'] for msg in self.history if msg['role'] == 'user'), "")
        if first_user_msg:
            title = first_user_msg[:max_length]
            if len(first_user_msg) > max_length:
                title += "..."
            return title
        return f"Chat - {datetime.now().strftime('%H:%M')}"

    def update_chat_history(self, history: List[Dict[str, Any]]):
        """Update chat history"""
        self.history = history.copy()
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    # def save_chat(self, chat_id: str, history: List[Dict], model: str, system_message: str):
    #     """Save chat to storage"""
    #     CHAT_STORAGE[chat_id] = {
    #         'history': history.copy(),
    #         'model': model,
    #         'system_message': system_message,
    #         'title': get_chat_title(history),
    #         'created_at': datetime.now().isoformat(),
    #         'updated_at': datetime.now().isoformat()
    #     }
    #
    # def load_chat(self, chat_id: str) -> Tuple[List[Dict], str, str]:
    #     """Load chat from storage"""
    #     if chat_id in CHAT_STORAGE:
    #         chat_data = CHAT_STORAGE[chat_id]
    #         return chat_data['history'], chat_data['model'], chat_data['system_message']
    #     return [], "", ""
    #
    # def get_chat_list(self) -> List[Tuple[str, str]]:
    #     """Get list of chats for dropdown"""
    #     chat_list = []
    #     for chat_id, chat_data in sorted(CHAT_STORAGE.items(),
    #                                      key=lambda x: x[1]['updated_at'],
    #                                      reverse=True):
    #         title = chat_data['title']
    #         chat_list.append((title, chat_id))
    #     return chat_list
    #
    # def delete_chat(chat_id: str):
    #     """Delete chat from storage"""
    #     if chat_id in CHAT_STORAGE:
    #         del CHAT_STORAGE[chat_id]
