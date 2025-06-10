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
from sympy import false
import random
import inspect

from rameshm.llmeng.llmchat.llm_chat import LlmChat
from rameshm.llmeng.utils.init_utils import set_environment_logger
from rameshm.llmeng.exception.llm_chat_exception import LlmChatException

logger = set_environment_logger()

def set_chat_selector_drop_down(chat_list: Dict[int, LlmChat], current_chat_id: Optional[int] = None) -> Dict:
    """
    Set the chat selector dropdown with chat names and IDs.

    Args:
        chat_list: Dictionary of LlmChat objects with chat_id as keys
        current_chat_id: Optional current chat ID to set as selected

    Returns:
        Gradio Dropdown component with updated choices
    """
    chat_nm_list = get_chat_nm_list(chat_list)
    if current_chat_id is not None and current_chat_id in chat_list:
        chat_list_drop_down = gr.update(choices=chat_nm_list, value=current_chat_id,label="📋 Select Chat")
    else:
        chat_list_drop_down = gr.update(choices=chat_nm_list, value=None, label="📋 Select Chat")
    return chat_list_drop_down

def extract_from_gr_state_with_type_check(state_obj: Any, expected_type: type, default: Any = None) -> Any:
    """
    Extract value from State and ensure it's the expected type.

    Args:
        state_obj: Gradio State object or other Python object
        expected_type: Expected type of the contained value
        default: Default value if type doesn't match or extraction fails

    Returns:
        The extracted value if it matches expected_type, otherwise default
    """
    print(f"DEBUG: Extracting {state_obj} of type {type(state_obj)} expecting {expected_type}")
    try:
        # Extract the value
        if isinstance(state_obj, gr.State):
            value = state_obj.value
        else:
            value = state_obj

        # Check type
        if isinstance(value, expected_type):
            return value
        else:
            return default

    except (AttributeError, TypeError):
        return default

def validate_inputs(message: str, history: List, model: str, system_message: str) -> tuple:
    """Validate inputs before processing"""
    if not message or not message.strip():
        logger.error("Please enter a message")
        return false, "Please enter a message"

    if not model:
        logger.error("Please enter a model")
        return False, "Please select a model"

    # Check if required API keys are available
    model_name = model.lower()
    #  Ollama runs locally, so no API key is needed
    if "gpt" in model_name and not os.getenv("OPENAI_API_KEY"):
        msg = "OpenAI API key not found in environment variables"
        logger.error(msg)
        return False, msg
    elif "claude" in model_name and not os.getenv("ANTHROPIC_API_KEY"):
        msg = "Anthropic API key not found in environment variables"
        logger.error(msg)
        return False, msg
    elif "gemini" in model_name and not os.getenv("GOOGLE_API_KEY"):
        msg = "Google API key not found in environment variables"
        logger.error(msg)
        return False, msg
    return True, ""

def get_model(model_nm: str):
    model_nm = model_nm.split(" ")[1].strip()
    print(f"Using Model: {model_nm}")
    if "gpt" in model_nm:
        return ChatOpenAI(model=model_nm, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, timeout=30)
    elif "claude" in model_nm:
        return ChatAnthropic(model=model_nm, api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30,
                             temperature=0.7, max_tokens=1024,top_p=0.9, top_k=40)
    elif "llama" in model_nm or "gemma" in model_nm:
        # Ollama run on "http://localhost:11434"  # Default Ollama URL. If you type that URL you shoud see "Ollama Running" message
        return Ollama(model=model_nm, # api_key="ollama",base_url="http://localhost:11434",
                      temperature=0.7, top_p=0.9, top_k=40, num_predict=256, repeat_penalty=1.1)
    elif "gemini" in model_nm:
         return ChatGoogleGenerativeAI(model=model_nm, google_api_key=os.getenv("GOOGLE_API_KEY"), timeout=30)
    else:
        raise LlmChatException("Model: {model_nm} is not supported")

def build_langchain_history(history: List, system_message: Optional[str]) -> List:
    """Build Langchain history from conversation history, system message, and user message"""
    langchain_history = []

    # Add system message if provided
    if system_message and system_message.strip():
        langchain_history.append(SystemMessage(content=system_message.strip()))
        logger.info(f"Added system message ({len(system_message)} chars)")

    # Add conversation history
    for i, msg in enumerate(history):
        try:
            if msg.get('role') == "user":
                langchain_history.append(HumanMessage(content=msg['content']))
            elif msg.get('role') == "assistant":
                langchain_history.append(AIMessage(content=msg['content']))
        except Exception as e:
            err_msg = f"Malformed message history at index {i}: {msg}"
            raise LlmChatException(err_msg) from e

    logger.debug(f"Total messages in Langchain History for API Call: {len(langchain_history)}")

    return langchain_history

def get_response_from_model(model, langchain_history: List) -> str:
    # Submit te request containing the conversation history and user message to get the model response
    start_time = time.time()
    llm_response = model.invoke(langchain_history)
    elapsed = time.time() - start_time
    logger.debug(f"API call completed in {elapsed:.2f}s")

    # Handle response format differences
    if isinstance(llm_response, str):
        response_content = llm_response
    else:
        response_content = getattr(llm_response, 'content', str(llm_response))

    if response_content:
        logger.debug(f"Received response: {response_content}\n")
        return response_content
    else:
        err_msg = "Received empty response from model: {selected_model} for message: {message}"
        raise LlmChatException(err_msg)


def generate_chat_id(chat_list: Dict[int, LlmChat]) -> int:
    """Generate a unique chat ID based on current timestamp"""
    if not chat_list and len(chat_list):
        max_chat_id = max(list(chat_list))
        return max_chat_id + 1
    return 1    # # Start with 1 if no chats exist


def create_update_chat_in_storage(current_chat_id: Optional[int], updated_history: List,
                                  chat_list: Dict[int, LlmChat], selected_model: str = None,
                                  system_message: str = None) -> (str, List[Dict[str, LlmChat]]):
    if not current_chat_id:
        logger.debug(f"Creating new chat with History: {updated_history}")
        current_chat_id = generate_chat_id(chat_list)
        current_llm_chat = LlmChat(chat_id=current_chat_id,
                                   model_nm=selected_model, # Model name can't be None, so it is not checked here
                                   history=updated_history,
                                   system_message=system_message if system_message else "")

        chat_list[current_chat_id] = current_llm_chat
    else:
        # Update current with updated chat
        current_llm_chat = chat_list[current_chat_id]
        current_llm_chat.update_chat_history(updated_history)
    logger.debug(f"Chat updated with ID: {current_chat_id}, History Length: {len(updated_history)}")
    return current_chat_id, chat_list

def get_chat_nm_list(chat_list: Any) -> List[tuple[str, int]]:
    """ Get list of Chat Names and ChatId. Reverse order to show latest chats first"""
    # This method is called directly either from the UI or from the predict function. Need to ensure that the chat_list is a dictionary
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})
    chat_list_sorted = {k: chat_list[k] for k in sorted(chat_list.keys())}
    for k, llm_chat in chat_list_sorted.items():
        print(f"****DEBUG 78f8j8**** Chat List Contents: Key: {k}, Title: {llm_chat.get_chat_title()}, Model: {llm_chat.get_model_nm()}")
    return [(llm_chat.get_chat_title(), k) for k, llm_chat in chat_list_sorted.items()]


def predict(message: str, history: List, selected_model: str, system_message: str,
            current_chat_id: Optional[int], chat_list: Dict[int, LlmChat]) -> Tuple[str, List, List, int, List, Dict]:
    """Enhanced predict function with chat management"""
    start_time = time.time()

    # Convert Gradio State objects to regular Python object types as needed
    history = extract_from_gr_state_with_type_check(history, list, [])
    current_chat_id = extract_from_gr_state_with_type_check(current_chat_id, int, None)
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})

    chat_list_updated = chat_list
    current_function_name = inspect.currentframe().f_code.co_name

    try:
        # Validate inputs
        is_valid, error_msg = validate_inputs(message, history, selected_model, system_message)
        if not is_valid:
            raise LlmChatException(error_msg)

        logger.debug(f"Processing request - Model: {selected_model}, Message: {message}")

        # Initialize model
        model = get_model(selected_model)
        logger.debug(f"Model initialized: {type(model).__name__}")

        # Build langchain history
        langchain_history = build_langchain_history(history, system_message)
        # Add current user message
        langchain_history.append(HumanMessage(content=message))

        # Make call to the model and get response
        response_content = get_response_from_model(model, langchain_history)

        if not response_content:
            err_msg = "Received empty response from model: {selected_model} for message: {message}"
            raise LlmChatException(err_msg)
        else:
            logger.info(f"Received response: {response_content}\n")

        # Update conversation history
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_content}
        ]

        # Save/update chat
        current_chat_id, chat_list = create_update_chat_in_storage(current_chat_id, history, chat_list,
                                                                   selected_model, system_message)

        # Construct the Dict for updating chat_list drop down.
        chat_list_drop_down = set_chat_selector_drop_down(chat_list, current_chat_id)
        #chat_nm_list = get_chat_nm_list(chat_list)
        #chat_list_drop_down = gr.update(choices=chat_nm_list, value=current_chat_id)

        elapsed = time.time() - start_time
        logger.debug(f"💬 Response: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")
        logger.debug(f"Elapsed timeL {elapsed:.2f}s in Function: {current_function_name} for model: {selected_model}")
        return "", history, history, current_chat_id, chat_list, chat_list_drop_down,
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed:.2f}s: {e}", exc_info=True)
        error_response = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"🔥 ERROR Exception: Unexpected error: {str(e)}"}
        ]
        updated_history = history + error_response

        # Construct the Dict for updating chat_list drop down. Pointing to the current chat_id
        chat_list_drop_down = set_chat_selector_drop_down(chat_list, current_chat_id)

        return "", updated_history, updated_history, current_chat_id or "", chat_list, chat_list_drop_down,


def start_new_chat(chat_list: Dict[int, LlmChat]) -> Tuple[List, List, str, str, Optional[int], Dict]:
    """Start a new chat"""
    logger.info("Started new chat")
    chat_list_drop_down = set_chat_selector_drop_down(chat_list)
    return [], [], "", "", None, chat_list_drop_down,
#outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector]


def load_selected_chat(chat_id: str, chat_list: Dict[int, LlmChat]) -> Tuple[List, List, str, str, str, int]:
    """Load selected chat from the Chat List"""

    # Convert Gradio State objects to regular Python object types as needed
    chat_id = extract_from_gr_state_with_type_check(chat_id, int, None)
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})

    chat_id = int(chat_id) if chat_id and chat_id.isdigit() else None
    #llm_chat = next((chat_dict[chat_id] for chat_dict in chat_list if chat_id in chat_dict), None)
    llm_chat = chat_list.get(chat_id, None)
    #TODO Should the ChatList be updated here to reflect the current chat?
    if chat_id and llm_chat:
        history = llm_chat.get_history()
        model = llm_chat.get_model_nm()
        system_message = llm_chat.system_message
        user_input = ""
        logger.info(f"Loaded chat: {chat_id} with model: {model}")
        return history, history, user_input, model, system_message, chat_id
    else:
        # Looks like ChatId is not valid or chat history is empty. This can happen when no chat is created yet.
        # Instead of raising an exception, we will return empty values to reset the UI.
        logger.warning(f"Chat ID {chat_id} not found or has no history. Resetting UI.")
        return [], [], "", "", "", chat_id


#TODO: It is cleaning out the chat history etc., even when the chat being deleted is not the current chat. Fix this so that
# it deletes the chat from the list only if we are deleting the current chat.
# Also a good idea to have a confirmation dialog before deleting the chat.
def delete_selected_chat(deleted_chat_id: str, chat_list: Dict[int, LlmChat], user_input, current_chat_id) -> Tuple[List, str, str, Optional[int], List[tuple[str, str]]]:
    """Delete selected chat"""

    # Convert Gradio State objects to regular Python object types as needed
    deleted_chat_id = extract_from_gr_state_with_type_check(deleted_chat_id, int, None)
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})
    current_chat_id = extract_from_gr_state_with_type_check(current_chat_id, int, None)

    deleted_chat_id = int(deleted_chat_id) if deleted_chat_id.isdigit() else None

    if deleted_chat_id and deleted_chat_id in chat_list:
        logger.info(f"Deleting chat ID: {deleted_chat_id}")
        # Remove chat from the list
        del chat_list[deleted_chat_id]
    else:
        logger.warning(f"Chat ID {deleted_chat_id} not found. No action taken.")
    if deleted_chat_id == current_chat_id or current_chat_id is None:
        current_chat_id = None
        return [], "", "", None, get_chat_nm_list(chat_list)
    else:
        logger.info(f"Chat ID {deleted_chat_id} deleted, but it was not the current chat. No action taken on current chat.")
        llm_chat = chat_list.get(current_chat_id, None)
        history = llm_chat.get_history()
        system_message = llm_chat.system_message
        return history, user_input, system_message, current_chat_id, get_chat_nm_list(chat_list)


# Launch configuration
# if __name__ == "__main__":
#     print("🚀 Starting Enhanced Multi-LLM Chatbot...")
#     print("📋 Features: Chat Management, File Upload, Clipboard Support")
#
#     multi_model_chat.launch(
#         inbrowser=True,
#         share=False,  # Set to True for public sharing
#         debug=False,  # Set to True for development
#         server_name="127.0.0.1",
#         server_port=7860
#     )