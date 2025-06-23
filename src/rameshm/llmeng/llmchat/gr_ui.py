"""
Multi-LLM Chatbot with Gradio UI. All Gradio components are defined here.
"""
import gradio as gr
import logging
from typing import List, Dict, Any, Tuple, Optional

from rameshm.llmeng.llmchat.llm_chat import LlmChat
from rameshm.llmeng.utils.init_utils import set_environment_logger
import rameshm.llmeng.llmchat.gr_event_handler as gr_event_handler
import rameshm.llmeng.llmchat.chat_constants as chat_constants

logger = set_environment_logger()

# Enhanced UI with chat management
#def create_gr_app():
# Define all Gradio components and their interactions

# Define Theme
my_theme = gr.themes.Default(
    font=["Times New Roman", "Times", "serif"],
    text_size=gr.themes.sizes.text_md
)

# Prepare model choices and a robust default
model_names = sorted([model['model_nm'] for model in chat_constants.MODEL_ATTRIBUTES])
# Ensure the default value is always valid
default_model = "llama3.2" if "llama3.2" in model_names else model_names[0] if model_names else None

# Create the Gradio app
with gr.Blocks(title="Multi-LLM Chatbot", theme=gr.themes.Soft()) as multi_model_chat:
    # State variables for chat management
    current_chat_id = gr.State(None)    # Store current chat ID
    chat_list = gr.State({})            # Store chat list as an array of dictionaries with key as chat_id and value as llm_chat
    chat_history = gr.State([])         # Store chat history as an array of dictionaries with role and content

    gr.Markdown("""
    # 🧠 Multi-LLM Chatbot with Chat Management""")
    # Chat with different AI models and manage your conversation history.
    # """)

    with gr.Row():
        # Left column for chat list
        with gr.Column(scale=1, min_width=250):
            # gr.Markdown("### 💬 Chat History")

            new_chat_btn = gr.Button(
                "➕ New Chat",
                variant="primary",
                size="sm"
            )

            chat_selector = gr.Dropdown(
                choices=gr_event_handler.get_chat_nm_list(chat_history),
                label="📋 Select Chat",
                interactive=True,
                allow_custom_value=False
            )

            delete_chat_btn = gr.Button(
                "🗑️ Delete Chat",
                variant="secondary",
                size="sm"
            )

            # gr.Markdown("---")
            # gr.Markdown("### ⚙️ Settings")

            model_selector = gr.Dropdown(
                choices=sorted([model['model_nm'] for model in chat_constants.MODEL_ATTRIBUTES]),
                value=default_model,
                label="🤖 LLM Model",
                interactive=True
            )

            system_message = gr.Textbox(
                placeholder="Enter system instructions...",
                label="📝 System Message",
                max_lines=3
            )

            # File upload component
            file_upload = gr.File(
                label="📎 File upload not supported for this model",
                file_types=None,
                file_count="multiple",
                interactive=False
            )

        # Right column for chat interface
        with gr.Column(scale=3):
            # Display model info
            # gr.Markdown("""
            # **💡 Features:**
            # - 💾 **Chat Management**: Save and load conversations
            # - 📎 **File Upload**: Upload images and text files
            # - 📋 **Clipboard**: Paste content directly into messages
            # - 🎯 **Multiple Models**: OpenAI, Claude, Gemini, Ollama
            # """)

            chatbot = gr.Chatbot(
                label="💬 Conversation",
                type="messages",
                height=450,
                show_copy_button=True,
                autoscroll=True
            )

            # with gr.Row():
            #     with gr.Column(scale=20):
            user_input = gr.Textbox(
                placeholder="Type your message here... You can paste text from clipboard!",
                label="✍️ Your Message",
                max_lines=5,
                show_copy_button=True
            )
                # The submit is now only through by clicking return on message input box.
                # with gr.Column(scale=1):
                #     send_btn = gr.Button(
                #         "📤 Send",
                #         variant="primary",
                #         size="lg"
                #     )
                #
                #     clear_btn = gr.Button(
                #         "🧹 Clear",
                #         variant="secondary",
                #         size="sm"
                #     )

    # Event handlers for chat management
    new_chat_btn.click(
        fn=gr_event_handler.start_new_chat,
        inputs=[chat_list, model_selector],
        outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector, file_upload]
    )

    model_selector.select(
        fn=gr_event_handler.start_new_chat,
        inputs=[chat_list, model_selector],
        outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector, file_upload]
    )

    system_message.submit(
        fn=gr_event_handler.start_new_chat,
        inputs=[chat_list, model_selector],
        outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector, file_upload]
    )

    chat_selector.select(
        fn=gr_event_handler.load_selected_chat,
        inputs=[chat_selector, chat_list],
        outputs=[chatbot, chat_history, user_input, model_selector, system_message, current_chat_id, chat_selector, file_upload]
    )

    delete_chat_btn.click(
        fn=gr_event_handler.delete_selected_chat,
        inputs=[chat_selector,chat_list, user_input, current_chat_id, model_selector],
        outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector, file_upload]
    )

    # File upload handler with proper image processing
    #uploaded_file_data = gr.State(None)


    user_input.submit(
        fn=lambda msg, hist, model, sys_msg, chat_id, chat_list, file_upload: gr_event_handler.predict(msg, hist, model, sys_msg,
                                                                                                       chat_id, chat_list, file_upload),
        inputs=[user_input, chat_history, model_selector, system_message, current_chat_id, chat_list, file_upload],
        outputs=[user_input, chat_history, chatbot, current_chat_id, chat_list, chat_selector, file_upload]
    )  #.then(
    #     lambda: None,  # Clear uploaded file data after sending
    #     outputs=[uploaded_file_data]
    # )

    # send_btn.click(
    #     fn=lambda msg, hist, model, sys_msg, chat_id, chat_list, file_upload: gr_event_handler.predict(msg, hist, model, sys_msg,
    #                                                                                       chat_id, chat_list, file_upload),
    #     inputs=[user_input, chat_history, model_selector, system_message, current_chat_id, chat_list, file_upload],
    #     outputs=[user_input, chat_history, chatbot, current_chat_id, chat_list, chat_selector]
    # ).then(
    #     lambda: None,  # Clear uploaded file data after sending
    #     outputs=[uploaded_file_data]
    # )

    # clear_btn.click(
    #     fn=gr_event_handler.start_new_chat,
    #     inputs=[chat_list],
    #     outputs=[chatbot, chat_history, user_input, system_message, current_chat_id, chat_selector]
    # )

if __name__ == "__main__":
    # Initialize the Gradio app
    try:
        multi_model_chat.close()
    except Exception as e:
        logger.error(f"Error in the App: {e}", exc_info=True)

    multi_model_chat.launch(inbrowser=True)
