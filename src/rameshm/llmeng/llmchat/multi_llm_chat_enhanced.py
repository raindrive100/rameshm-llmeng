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

from rameshm.llmeng.llmchat.llm_chat import LlmChat
from rameshm.llmeng.utils.init_utils import set_environment_logger

logger = set_environment_logger()

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
        raise Exception("Model: {model_nm} is not supported")

def build_langchain_history(history: List, system_message, user_message: str) -> List:
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
            logger.error(err_msg)
            raise ValueError(err_msg) from e

        # Add current user message
        langchain_history.append(HumanMessage(content=message))

        logger.debug(f"Built conversation with {len(langchain_history)} messages")
        logger.debug(f"Making API call with model: {selected_model} with total messages: {len(langchain_history)}")

    return langchain_history

def get_response_from_model(model, langchain_history: List) -> str:
    # Submit te request containing the conversation history and user message to get the model response
    llm_response = model.invoke(langchain_history)
    elapsed = time.time() - start_time
    logger.debug(f"API call completed in {elapsed:.2f}s")

    # Handle response format differences
    if isinstance(llm_response, str):
        response_content = llm_response
    else:
        response_content = getattr(llm_response, 'content', str(llm_response))

    if not response_content:
        err_msg = "Received empty response from model: {selected_model} for message: {message}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    else:
        logger.debug(f"Received response: {response_content}\n")

def generate_chat_id() -> str:
    """Generate a unique chat ID based on current timestamp"""
    return f"chat_{int(time.time())}_{random.randint(1,10000)}"

def create_update_chat_in_storage(current_chat_id: Optional[str], updated_history: List,
                                  chat_list: List[Dict[str, LlmChat]]) -> (str, List[Dict[str, LlmChat]]):
    if not current_chat_id:
        current_chat_id = generate_chat_id()
        current_llm_chat = LlmChat(chat_id=current_chat_id, history=updated_history)
        chat_list += [{current_chat_id: current_llm_chat}]
    else:
        # Update current with updated chat
        current_llm_chat = next((chat_dict[current_chat_id] for chat_dict in chat_list if current_chat_id in chat_list), None)
        current_llm_chat.update_chat_history(updated_history)
    return current_chat_id, chat_list

def get_chat_nm_list(chat_list: List[Dict[str, LlmChat]]) -> List[tuple[str, str]]:
    """ Get list of Chat Names and ChatId. Reverse order to show latest chats first"""
    return [(llm_chat.get_chat_title(), llm_chat.get_chat_id())
            for chat in chat_list
            for chat_id, llm_chat in chat.items()][::-1]  # Reverse order to show latest chats first

def predict(message: str, history: List, selected_model: str, system_message: str,
            current_chat_id: Optional[str], chat_list: List[Dict[str, LlmChat]]) -> Tuple[str, List, List, str, List, List]:
    """Enhanced predict function with chat management"""
    start_time = time.time()
    chat_list_updated = chat_list

    try:
        # Validate inputs
        is_valid, error_msg = validate_inputs(message, history, selected_model, system_message)
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            raise ValueError(error_msg)

        logger.debug(f"Processing request - Model: {selected_model}, Message: {message}")

        # Ensure history is a list
        if not isinstance(history, list):
            history = history.value if hasattr(history, 'value') else []

        # Initialize model
        model = get_model(selected_model)
        logger.debug(f"Model initialized: {type(model).__name__}")

        # Build langchain history
        langchain_history = build_langchain_history(history, system_message)

        # Make call to the model and get response
        response_content = get_response_from_model(model, langchain_history)
        llm_response = model.invoke(langchain_history)
        elapsed = time.time() - start_time
        logger.debug(f"API call completed in {elapsed:.2f}s")

        if not response_content:
            err_msg = "Received empty response from model: {selected_model} for message: {message}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info(f"Received response: {response_content}\n")

        # Update conversation history
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_content}
        ]

        # Save/update chat
        chat_id, chat_list_updated = create_update_chat_in_storage(current_chat_id, updated_history, chat_list)

        logger.debug(f"💬 Response: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")
        return "", updated_history, updated_history, current_chat_id, chat_list_updated, get_chat_nm_list(chat_list_updated),
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed:.2f}s: {e}")
        error_response = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"🔥 ERROR Exception: Unexpected error: {str(e)}"}
        ]
        updated_history = history + error_response
        return "", updated_history, updated_history, current_chat_id or "", chat_list_updated, get_chat_nm_list(chat_list_updated),


def start_new_chat():
    """Start a new chat"""
    logger.info("Started new chat")
    return [], "", "", None, "",


def load_selected_chat(chat_id: str, chat_list: List[Dict[str, LlmChat]]) -> Tuple[List, List, str, str, str, str]:
    """Load selected chat from the list"""
    llm_chat = next((chat_dict[chat_id] for chat_dict in chat_list if chat_id in chat_list), None)
    history = llm_chat.get_history()
    model = llm_chat.get_model_nm()
    system_message = llm_chat.system_message
    user_input = ""
    logger.info(f"Loaded chat: {chat_id} with model: {model}")
    return history, history, user_input, model, system_message, chat_id



#TODO: It is cleaning out the chat history etc., even when the chat being deleted is not the current chat. Fix this so that
# it deletes the chat from the list only if we are deleting the current chat.
# Also a good idea to have a confirmation dialog before deleting the chat.
def delete_selected_chat(chat_id: str, chat_list: List[Dict[str, LlmChat]]) -> Tuple[List, str, str, Optional[str], List[tuple[str, str]]]:
    """Delete selected chat"""
    chat_list = [chat_dict for chat_dict in chat_list if chat_id not in chat_dict]
    return [], "", "", "", get_chat_nm_list(chat_list)


# Enhanced UI with chat management
with gr.Blocks(title="Multi-LLM Chatbot", theme=gr.themes.Soft()) as multi_model_chat:
    # State variables for chat management
    current_chat_id = gr.State(None)
    chat_list = gr.State([])    # Store chat list as an array of dictionaries with key as chat_id and value as llm_chat
    chat_history = gr.State([]) # Store chat history as an array of dictionaries with role and content

    gr.Markdown("""
    # 🧠 Multi-LLM Chatbot with Chat Management

    Chat with different AI models and manage your conversation history.
    """)

    with gr.Row():
        # Left column for chat list
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 💬 Chat History")

            new_chat_btn = gr.Button(
                "➕ New Chat",
                variant="primary",
                size="sm"
            )

            chat_selector = gr.Dropdown(
                choices=get_chat_nm_list(chat_history),
                label="📋 Select Chat",
                interactive=True,
                allow_custom_value=False
            )

            delete_chat_btn = gr.Button(
                "🗑️ Delete Chat",
                variant="secondary",
                size="sm"
            )

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Settings")

            model_selector = gr.Dropdown(
                choices=[
                    "OpenAI: gpt-4o",
                    "OpenAI: gpt-4o-mini",
                    "OpenAI: gpt-3.5-turbo",
                    "Claude: claude-3-5-sonnet-20241022",
                    "Claude: claude-3-5-haiku-20241022",
                    "Google: gemini-1.5-flash",
                    "Google: gemini-1.5-pro",
                    "Ollama: llama3.2",
                    "Ollama: gemma2:2b",
                    "Ollama: qwen2.5:3b",
                    "Ollama: mistral:7b"
                ],
                value="OpenAI: gpt-4o-mini",
                label="🤖 AI Model",
                interactive=True
            )

            system_message = gr.Textbox(
                placeholder="Enter system instructions...",
                label="📝 System Message",
                max_lines=3
            )

        # Right column for chat interface
        with gr.Column(scale=3):
            # Display model info
            gr.Markdown("""
            **💡 Features:**
            - 💾 **Chat Management**: Save and load conversations
            - 📎 **File Upload**: Upload images and text files  
            - 📋 **Clipboard**: Paste content directly into messages
            - 🎯 **Multiple Models**: OpenAI, Claude, Gemini, Ollama
            """)

            chatbot = gr.Chatbot(
                label="💬 Conversation",
                type="messages",
                height=400,
                show_copy_button=True
            )

            with gr.Row():
                with gr.Column(scale=4):
                    user_input = gr.Textbox(
                        placeholder="Type your message here... You can paste images or text from clipboard!",
                        label="✍️ Your Message",
                        max_lines=5,
                        show_copy_button=True
                    )

                    # File upload component
                    file_upload = gr.File(
                        label="📎 Upload File (Images, Text, Code)",
                        file_types=["image", ".txt", ".md", ".py", ".js", ".html", ".css", ".json"]
                    )

                with gr.Column(scale=1):
                    send_btn = gr.Button(
                        "📤 Send",
                        variant="primary",
                        size="lg"
                    )

                    clear_btn = gr.Button(
                        "🧹 Clear",
                        variant="secondary",
                        size="sm"
                    )

    # Event handlers for chat management
    new_chat_btn.click(
        fn=start_new_chat,
        outputs=[chatbot, user_input, system_message, current_chat_id, chat_selector]
    )

    chat_selector.change(
        fn=load_selected_chat,
        inputs=[chat_selector, chat_list],
        outputs=[chat_history, chatbot, user_input, model_selector, system_message, current_chat_id]
    )

    delete_chat_btn.click(
        fn=delete_selected_chat,
        inputs=[chat_selector,chat_list],
        outputs=[chatbot, user_input, system_message, current_chat_id, chat_selector]
    )

    # File upload handler with proper image processing
    uploaded_file_data = gr.State(None)


    def handle_file_upload(file_path, current_message):
        if file_path:
            text_content, image_base64, image_format = process_file_upload(file_path)

            # Store the processed file data
            file_data = (text_content, image_base64, image_format) if image_base64 else (text_content, None, None)

            # Update the message textbox
            if current_message:
                new_message = f"{current_message}\n\n{text_content}" if text_content else current_message
            else:
                new_message = text_content or ""

            return new_message, file_data
        return current_message, None


    file_upload.change(
        fn=handle_file_upload,
        inputs=[file_upload, user_input],
        outputs=[user_input, uploaded_file_data]
    )

    # Main prediction handlers with file upload support
    user_input.submit(
        fn=lambda msg, hist, model, sys_msg, chat_id, file_data: predict(msg, hist, model, sys_msg, chat_id, file_data),
        inputs=[user_input, chat_history, model_selector, system_message, current_chat_id, chat_list, uploaded_file_data],
        outputs=[user_input, chat_history, chatbot, current_chat_id, chat_list, chat_selector]
    ).then(
        lambda: None,  # Clear uploaded file data after sending
        outputs=[uploaded_file_data]
    )

    send_btn.click(
        fn=lambda msg, hist, model, sys_msg, chat_id, file_data: predict(msg, hist, model, sys_msg, chat_id, file_data),
        inputs=[user_input, chat_history, model_selector, system_message, current_chat_id, uploaded_file_data],
        outputs=[user_input, chat_history, chatbot, current_chat_id, chat_selector]
    ).then(
        lambda: None,  # Clear uploaded file data after sending
        outputs=[uploaded_file_data]
    )

    clear_btn.click(
        fn=start_new_chat,
        outputs=[chatbot, user_input, system_message, current_chat_id, chat_selector]
    )

# Launch configuration
if __name__ == "__main__":
    print("🚀 Starting Enhanced Multi-LLM Chatbot...")
    print("📋 Features: Chat Management, File Upload, Clipboard Support")

    multi_model_chat.launch(
        inbrowser=True,
        share=False,  # Set to True for public sharing
        debug=False,  # Set to True for development
        server_name="127.0.0.1",
        server_port=7860
    )