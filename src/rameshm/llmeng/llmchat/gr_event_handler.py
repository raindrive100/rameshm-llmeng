# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os
import time
from typing import List, Dict, Any, Tuple, Optional

from rameshm.llmeng.llmchat.llm_chat import LlmChat
from rameshm.llmeng.utils.init_utils import set_environment_logger
from rameshm.llmeng.exception.llm_chat_exception import LlmChatException
from rameshm.llmeng.llmchat.file_handler_llm import FileToLLMConverter
import rameshm.llmeng.llmchat.chat_constants as chat_constants


# Set the logger
logger = set_environment_logger()

def set_chat_selector_drop_down(chat_list: Dict[int, LlmChat], current_chat_id: Optional[int] = None) -> gr.Dropdown:
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
        #chat_list_drop_down = gr.update(choices=chat_nm_list, value=current_chat_id,label="📋 Select Chat")
        chat_list_drop_down = gr.Dropdown(choices=chat_nm_list, value=current_chat_id)
    else:
        #chat_list_drop_down = gr.update(choices=chat_nm_list, value=None, label="📋 Select Chat")
        chat_list_drop_down = gr.Dropdown(choices=chat_nm_list, value=None)
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
    logger.debug(f"DEBUG: Extracting state object of type: {type(state_obj)} expecting {expected_type}")

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

def validate_inputs(message: str, history: List, model: str, system_message: str) -> tuple:
    """Validate inputs before processing"""
    if not message or not message.strip():
        logger.error("User message is empty. Enter a message.")
        return False, "Enter a message"

    if not model:
        logger.error("Model must be selected. Please select a model.")
        return False, "Model must be selected. Please select a model."

    # Map model keywords to their required API key environment variables
    # Ollama is local and doesn't require a key, so it's omitted.
    api_key_requirements = {
        "gpt": ("OPENAI_API_KEY", "OpenAI"),
        "claude": ("ANTHROPIC_API_KEY", "Anthropic"),
        "gemini": ("GOOGLE_API_KEY", "Google"),
        #"llama": ("OLLAMA_API_KEY", "Ollama"),  # Ollama runs locally, so key is not need
    }

    model_name_lower = model.lower()
    # Ollama runs locally, so no API key is needed
    for keyword, (env_var, provider_name) in api_key_requirements.items():
        if keyword in model_name_lower and not os.getenv(env_var):
            msg = f"{provider_name} API key-{env_var} not found in environment variables"
            logger.error(msg)
            return False, msg

    return True, ""


def get_model(model_nm: str):
    logger.debug(f"Generating model instance for model: {model_nm}")
    max_output_tokens = chat_constants.MAX_LLM_OUTPUT_TOKENS  # Default max output tokens
    if "gpt" in model_nm:
        return ChatOpenAI(model=model_nm, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, timeout=30,
                          max_tokens=max_output_tokens)
    elif "claude" in model_nm:
        return ChatAnthropic(model=model_nm, api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30,
                             temperature=0.7, max_tokens=max_output_tokens,top_p=0.9, top_k=40)
    elif "llama" in model_nm or "gemma" in model_nm:
        # Ollama run on "http://localhost:11434"  # Default Ollama URL. If you type that URL you shoud see "Ollama Running" message
        # When built using Docker Compose Ollama runs as its own service and sets an environment OLLAMA_HOST. If that isn't defined
        # then default to localhost.
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return Ollama(model=model_nm, # api_key="ollama",base_url="http://localhost:11434",
                      base_url=ollama_host, temperature=0.7, num_predict=max_output_tokens,
                      top_p=0.9, top_k=40, repeat_penalty=1.1
                      )
    elif "gemini" in model_nm:
         return ChatGoogleGenerativeAI(model=model_nm, google_api_key=os.getenv("GOOGLE_API_KEY"), timeout=30,
                                       temperature=0.7, max_output_tokens=max_output_tokens,
                                       top_p=0.9, top_k=40
                                       )
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
                langchain_history.append(HumanMessage(content=msg['content_llm']))
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
    logger.debug(f"Invoking model: {type(model).__name__} with {len(langchain_history)} messages")
    llm_response = model.invoke(langchain_history)
    elapsed = time.time() - start_time
    logger.debug(f"Model API call completed in {elapsed:.2f}s")

    # Handle response format differences
    if isinstance(llm_response, str):
        response_content = llm_response
    else:
        response_content = getattr(llm_response, 'content', str(llm_response))

    if response_content:
        #logger.debug(f"Received response: {response_content}\n")
        logger.debug(f"Received non-zero response from model")
        return response_content
    else:
        err_msg = "Received empty response from model: {selected_model} for message: {message}"
        logger.error(err_msg)
        raise LlmChatException(err_msg)


def generate_chat_id(chat_list: Dict[int, LlmChat]) -> int:
    """Generate a unique chat ID based on current timestamp"""
    if chat_list and len(chat_list):
        max_chat_id = max(list(chat_list))
        logger.debug(f"Incrementing chat_id by 1. New chat_id is: {max_chat_id+1} ")
        return max_chat_id + 1
    else:
        logger.debug(f"First chat being created with chat_id = 1")
        return 1    # # Start with 1 if no chats exist


def create_update_chat_in_storage(current_chat_id: Optional[int], updated_history: List,
                                  chat_list: Dict[int, LlmChat], selected_model: str = None,
                                  system_message: str = None) -> (str, List[Dict[str, LlmChat]]):
    if not current_chat_id:
        # New chat. Create a new chat_id
        logger.debug(f"Creating new chat with History length of: {len(updated_history)}")
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
    logger.debug(f"Chat updated with ID: {current_chat_id}, Chat List Length: {len(chat_list)} History Length: {len(updated_history)}")
    return current_chat_id, chat_list

def get_chat_nm_list(chat_list: Any) -> List[tuple[str, int]]:
    """ Get list of Chat Names and ChatId. Reverse order to show latest chats first"""
    # This method is called directly either from the UI or from the predict function. Need to ensure that the chat_list is a dictionary
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})
    chat_list_sorted = {k: chat_list[k] for k in sorted(chat_list.keys(), reverse=True)} # Latest chat listed first
    return [(llm_chat.get_chat_title(), k) for k, llm_chat in chat_list_sorted.items()]


def validate_uploaded_files(file_paths_uploaded: List[str]) -> Tuple[List[str], bool, bool, str]:
    """Receive a list of files that are uploaded to be sent to LLM. Validates those files and return a list
    that are good to be processed or exception if there are serious issues that stops execution.

    Inputs:
        file_paths: A list of file paths that the user has uploaded
    Outputs:
        File Path List: List of files that passed all validations
        Boolean: True if validations are successful
        Boolean: True if Fatal Error (further processing should stop). False if not fatal error
        err_msg: Any error message from validation
    """
    logger.debug("Starting validate_uploaded_files")
    file_handler_llm = FileToLLMConverter()
    max_size = chat_constants.get_max_combined_size_of_files_upload()
    included_files = [f for f in file_paths_uploaded if os.path.isfile(f)]
    total_size = sum(os.path.getsize(f) for f in included_files)
    err_msgs = []
    fatal_error = False

    if total_size > max_size:
        msg = f"FATAL ERROR: Total size {total_size} exceeds max allowed {max_size}. Delete some files."
        logger.warning(msg)
        return [], True, True, msg

    valid_files = []
    skipped_file_cnt = 0
    for file_path in included_files:
        is_valid, err_msg = file_handler_llm.is_valid_file(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            skipped_file_cnt += 1
            logger.warning(f"Skipping file {file_path}: {err_msg}")
            err_msgs.append(f"{file_path}: {err_msg}")

    logger.debug(f"After validation following files are being included for sending to LLM: {valid_files}")
    logger.info(f"Out of a total of {len(file_paths_uploaded)} uploaded, "
                f"skipping: {skipped_file_cnt}, Sending: {len(valid_files)} files are being sent to LLM.")
    logger.debug("Finished validate_uploaded_files")

    return valid_files, False, False, "\n".join(err_msgs)


def model_supports_file_attachments(model_nm: str) -> bool:
    """
    Returns whether a model supports file attachments or not
    :param model_nm: name of the model
    :return: Returns true if the model is multi-model (i.e. can sypport text, image, voice etc.) else false
    """
    return model_nm in chat_constants.get_models_supporting_file_attachments()


def build_content_from_uploaded_files(file_paths_validated: List[str],
                                     check_file_validity: bool = True,
                                     include_images_in_pdf: bool = False,
                                     include_image_files: bool = False) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Returns a dictionary of the content
    :param file_paths_validated: List of files with path that need to be uploaded to LLM
    :param model_nm: Model Name to be used
    :param check_file_validity: Set to True if each file in file list should be checked against established criteria
    :param include_images_in_pdf: Set to True if images in PDF file should be included. "False" includes only text in PDF
    :param include_image_files: True-> Include contents of image file;False -> Image files are not sent to LLM
    :return: List of Dictionaries with File Data
             List of files that are successfully processed. Empty list if none of the files could be processed.
    """
    dropped_files = []
    file_contents = []

    logger.debug(f"Started building content of message for Validated file count of: {len(file_paths_validated)}")
    file_handler_llm = FileToLLMConverter()
    for index, file_path in enumerate(file_paths_validated):
        file_nm = os.path.basename(file_path)
        logger.debug(f"Started building content of message for file: {file_path}")
        file_data_text, file_meta_data = file_handler_llm.convert_file_to_str(file_path, check_file_validity,
                                                                              include_images_in_pdf)
        # Print first characters of the file content
        logger.debug(f"Generated text length of file: {len(file_data_text)}; "
                     f"First few char representation of file: {file_data_text[:100]}")
        if file_data_text:
            file_data_text = file_data_text.strip()
            file_type = file_meta_data["file_type"] # Has keys from chat_constants.SUPPORTED_FILE_TYPES
            mime_type = file_meta_data["mime_type"] # mime type from mimetypes package for a given file name.
            logger.debug(f"Building content dict for file_type: {file_type} and mime_type: {mime_type}")
            match str(file_type):
                case "text":
                    content_dict = {"type": "text", "text": file_data_text}
                    file_contents.append(content_dict)
                case "pdf":
                    if include_images_in_pdf:
                        content_dict = {
                        "type": "file", # Using Langchain which needs type to be "file". If using Anthropic directly then this should be "document"
                        "source_type": "base64",
                        "mime_type": "application/pdf",
                        "data": file_data_text,
                        }
                    else:
                        content_dict = {"type": "text", "text": file_data_text}
                    file_contents.append(content_dict)
                case "image":
                    if include_image_files:
                            content_dict = {
                            "type": "image",
                            "source_type": "base64",
                            "mime_type": "image/jpeg",
                            "data": file_data_text,
                            }
                            file_contents.append(content_dict)
                    else:
                        dropped_files.append(file_path)
                        logger.info(f"File is being dropped because it not a supported file type. File being skipped: {file_nm}")
                case "ms-wordx":
                    content_dict = {"type": "text", "text": file_data_text}
                    file_contents.append(content_dict)
                case None | _:
                    err_msg = f"Unsupported file type: {file_type} for File: {file_nm}. File is skipped"
                    dropped_files.append(file_path)
                    logger.warning(err_msg)
        logger.debug(f"Finished building content of message for file: {file_path}")

    processed_files = [file_path for file_path in file_paths_validated if file_path not in dropped_files]
    logger.info(f"Finished preparing content for all files. Excluded: {len(dropped_files)} out "
                f"of {len(file_paths_validated)} validated files")

    return file_contents, processed_files


def process_uploaded_files(file_paths_uploaded: List[str], model_nm: str) -> Tuple[List[Dict[str, str]],
                                                                             List[str]]:
    """
    Returns a dictionary of the content
    :param file_paths_uploaded: List of files with path that need to be uploaded to LLM
    :param model_nm: Model Name being used
    :return: List of Dictionaries with File Data or Empty list if none of the files are processed
             List of Strings with files that are Successfully processed. Empty if all None are successfully processed.
    """
    logger.debug("Starting reading of files uploaded")
    # Set the flags to include images if the model supports images.
    include_image_files = model_nm in chat_constants.get_image_handling_models()
    include_images_in_pdf = model_nm in chat_constants.get_pdf_image_handling_models()

    # Get filtered list that pass all the file checks and generate content only if model supports multi_modal
    if model_supports_file_attachments(model_nm) and len(file_paths_uploaded) > 0:
        included_files, validation_successful, fatal_error, err_msg_validation = validate_uploaded_files(
            file_paths_uploaded)
        if fatal_error:
            # Stop processing.
            logger.warning(err_msg_validation)
            raise LlmChatException(err_msg_validation)
        elif len(included_files) > 0:
            files_content, processed_files = build_content_from_uploaded_files(file_paths_validated=included_files,
                                                              check_file_validity = False, # Files already validated
                                                              include_images_in_pdf=include_images_in_pdf,
                                                              include_image_files=include_image_files
                                                              )
            return files_content, processed_files
        else:
            err_msg = (f"None of the files [{len(file_paths_uploaded)}] uploaded meet the required criteria. "
                       f"All files are skipped")
            logger.warning(err_msg)
            # Return empty lists and the caller to check and validate.
            return [], []
    else:
        # Model is not multi-modal or no files to process
        if model_supports_file_attachments(model_nm) and len(file_paths_uploaded) == 0:
            # Not an error
            logger.debug(f"Model: {model_nm} is multi-modal but zero files uploaded. Not an error")
            return [], []
        elif len(file_paths_uploaded) > 0 and not model_supports_file_attachments(model_nm):
            err_msg = f"Model: {model_nm} is NOT multi-modal, hence uploaded files are Ignored"
            logger.warning(err_msg)
            raise LlmChatException(err_msg)
        # Return empty lists and the caller to check and validate
        else:
            # Unexpected condition. Should never happen. More as safety check in case something was missed.
            raise LlmChatException("Unexpected Condition in if/else. Investigate")



def predict(message: str, history: List, selected_model: str, system_message: str,
            current_chat_id: Optional[int], chat_list: Dict[int, LlmChat],
            file_paths_uploaded: List[str]) -> Tuple[str, List, List, int, List, gr.Dropdown, List]:
    """Enhanced predict function with chat management"""
    logger.debug(f"Processing request - Model: {selected_model}, Message: {message}")
    start_time = time.time()

    # Convert Gradio State objects to regular Python object types as needed
    history = extract_from_gr_state_with_type_check(history, list, [])
    current_chat_id = extract_from_gr_state_with_type_check(current_chat_id, int, None)
    chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})

    # To avoid type failures. If file_paths were not uploaded then create an empty list
    file_paths_uploaded = [] if not file_paths_uploaded else file_paths_uploaded
    model_nm = selected_model
    try:
        # Validate inputs
        is_valid, err_msg = validate_inputs(message, history, selected_model, system_message)
        if not is_valid:
            logger.error(f"Invalid input: {err_msg}")
            raise LlmChatException(err_msg)

        # Initialize model
        model = get_model(model_nm)
        logger.debug(f"Model initialized: {type(model).__name__}")

        uploaded_file_content = []
        processed_files = []
        ok_to_attach_files = model_supports_file_attachments(model_nm)
        if ok_to_attach_files and file_paths_uploaded:
            uploaded_file_content, processed_files = process_uploaded_files(file_paths_uploaded, model_nm)
            not_processed_files = [file_path for file_path in file_paths_uploaded if file_path not in processed_files]
            if not_processed_files:
                skipped_files = [os.path.basename(file_path) for file_path in not_processed_files]
                err_file = "\n".join(skipped_files)
                err_msg = (f"Following files were not processed: {err_file}. Please check the following: "
                           f"1) File contents and file extensions match 2) Total size of files uploaded is less then 20MB "
                           f"3) That the files exist. Resubmit once corrected")
                logger.warning(err_msg)
                raise LlmChatException(err_msg)

        # Build langchain history
        langchain_history = build_langchain_history(history, system_message)
        # Add current user message. Message structure differs if model is multi-modal or not

        if ok_to_attach_files:
            message_content_for_llm = [{"type": "text", "text": message}] + uploaded_file_content
        else:
            message_content_for_llm = message
        langchain_history.append(HumanMessage(content=message_content_for_llm))
        logger.debug(f"Role: user message: {message[:300]}"
                     f"{'...' if len(message) > 300 else ''}") # print only small part of message

        # Make call to the model and get response
        response_content = get_response_from_model(model, langchain_history)

        if not response_content:
            err_msg = f"Received empty response from model: {model_nm} for message: {message}"
            logger.error(err_msg)
            raise LlmChatException(err_msg)
        else:
            logger.info(f"Response: {response_content[:300]}{'...' if len(response_content) > 300 else ''}")

        # Update conversation history
        history = history + [
            {"role": "user", "content": message, "content_llm": message_content_for_llm},
            {"role": "assistant", "content": response_content}
        ]

        # Save/update chat
        current_chat_id, chat_list = create_update_chat_in_storage(current_chat_id, history, chat_list,
                                                                   selected_model, system_message)
        logger.debug(f"Current chat_id: {current_chat_id} and length of chat_list: {len(chat_list)}")
        # Construct the Dict for updating chat_list drop down.
        chat_list_drop_down = set_chat_selector_drop_down(chat_list, current_chat_id)

        elapsed = time.time() - start_time
        logger.debug(f"Elapsed time in entire predict function "
                     f"{elapsed:.2f}s for model: {selected_model}")
        return "", history, history, current_chat_id, chat_list, chat_list_drop_down, [],
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed:.2f}s: {e}", exc_info=True)
        error_response = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"🔥 Please correct the following error and resubmit. ERROR: {str(e)}\n\n "}
        ]
        updated_history = history + error_response

        # Construct the Dict for updating chat_list drop down. Pointing to the current chat_id
        chat_list_drop_down = set_chat_selector_drop_down(chat_list, current_chat_id)
        #file_upload_object = get_file_upload_object(model_nm)

        # When an error happens we want to keep history without the error so that the chat can continue
        # Files uploaded are cleared.
        return "", history, updated_history, current_chat_id or "", chat_list, chat_list_drop_down, [],


def get_file_upload_object(model_nm: str = None) -> gr.File:
    """
    File Upload button accepts different file types depending on the model selected. This method returns a file object which can be used to
    set the file_upload widget on front end.

    :param model_nm: Model Name selected
    :return: returns the gr.File object that is used for setting the file_upload object
    """
    logger.debug(f"Setting file upload Object for model: {model_nm}")
    if model_nm.strip():
        # Set file-upload to the right set of files that can be uploaded and appropriately set the label to indicate the same
        file_types_supported = chat_constants.get_model_supported_file_types(model_nm)
        file_upload_label = "Upload Files: " + chat_constants.get_model_attributes(model_nm)

        if len(file_types_supported) > 0:
            file_upload = gr.File(label=file_upload_label, file_types= file_types_supported, interactive=True, value=None)
        else:
            # File attachments aren't supported. Disable the file_upload for inputs.
            file_upload = gr.File(label="File Upload Not Supported", file_types=None, interactive=False, value=None)
    else:
        # This should never happen. Just in case it happens disable file_upload to avoid issues
        file_upload = gr.File(label="File Upload Not Supported", file_types=None, interactive=False, value=None)

    logger.debug(f"Finished setting file upload Object for model: {model_nm}")
    return file_upload


def start_new_chat(chat_list: Dict[int, LlmChat], model_selected: str) -> Tuple[List, List, str, str, Optional[int], gr.Dropdown, gr.File]:
    """Starts a new chat
    This method resets the chatbot, history, user input, current chat ID, and system message.
    This is called from multiple places in the UI such as the New Chat button, Model or System Message changed.
    """
    logger.info(f"Started new chat for Model: {model_selected}")
    chat_list_drop_down = set_chat_selector_drop_down(chat_list)

    file_upload_object = get_file_upload_object(model_selected)
    # New chat means resetting the chatbot, history, user input, current chat ID, and system message
    return [], [], "", "", None, chat_list_drop_down, file_upload_object,


def load_selected_chat(chat_id: str, chat_list: Dict[int, LlmChat]) -> Tuple[List, List, str, str, str, int, gr.Dropdown, gr.File]:
    """Load selected chat from the Chat List"""
    # Error response to be displayed in the Chat area if chat is not found or unexpected issue happens.
    err_response = [
        {"role": "assistant",
         "content": f"🔥 ERROR Exception: Chat not found. Please select another chat or start a new one."
                    f"\n For Support folks: chat_id: {chat_id}"}
    ]
    try:
        logger.info(f"Loading chat_id: {chat_id}")
        # Convert Gradio State objects to regular Python object types as needed
        chat_id = extract_from_gr_state_with_type_check(chat_id, int, None)
        chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})

        llm_chat = chat_list.get(chat_id, None)

        if chat_id and llm_chat:
            history = llm_chat.get_history()
            model = llm_chat.get_model_nm()
            system_message = llm_chat.system_message
            user_input = ""
            logger.info(f"Loaded chat: {chat_id} with model: {model}")
            return history, history, user_input, model, system_message, chat_id, set_chat_selector_drop_down(chat_list, chat_id), get_file_upload_object(model),
        else:
            # Looks like ChatId is not valid or chat history is empty. This should never happen.
            # Instead of raising an exception, handle gracefully by returning empty values to reset the UI.
            logger.warning(f"chat_id {chat_id} not found or has no history. Resetting UI.")
            return err_response, [], "", "", "", chat_id, set_chat_selector_drop_down(chat_list, None), get_file_upload_object(None),
    except Exception as e:
        # Just in the rare case exception happens. Just reset things so that user can retry.
        err_msg = f"Error loading chat_id: {chat_id}: {str(e)}"
        logger.error(err_msg, exc_info=True)
        return err_response, [], "", "", "", chat_id, set_chat_selector_drop_down(chat_list, None), get_file_upload_object(None),
        #raise LlmChatException(err_msg) from e


# TODO: Good idea to have a confirmation dialog before deleting the chat.
def delete_selected_chat(delete_chat_id: str, chat_list: Dict[int, LlmChat], user_input, current_chat_id,
                         model_selected: str) -> Tuple[List, List, str, str, Optional[int], gr.Dropdown, gr.File]:
    """Delete selected chat"""
    logger.info(f"Deleting chat ID: {delete_chat_id}")
    # Error response to be displayed in the Chat area if chat is not found or unexpected issue happens.
    err_response = [
        {"role": "assistant",
         "content": f"🔥 ERROR Exception: Could not delete chat. Please select another chat or start a new one."
                    f"\n For Support folks: chat_id: {delete_chat_id}"}
    ]
    try:
        # Convert Gradio State objects to regular Python object types as needed
        delete_chat_id = extract_from_gr_state_with_type_check(delete_chat_id, int, None)
        chat_list = extract_from_gr_state_with_type_check(chat_list, dict, {})
        current_chat_id = extract_from_gr_state_with_type_check(current_chat_id, int, None)

        if delete_chat_id and delete_chat_id in chat_list:
            # Remove chat from the list
            del chat_list[delete_chat_id]
        else:
            logger.warning(f"chat_id: {delete_chat_id} to be deleted is not found. No action taken.")
        if delete_chat_id == current_chat_id or current_chat_id is None:
            current_chat_id = None
            return ([], [], "", "", None, set_chat_selector_drop_down(chat_list, current_chat_id),
                    get_file_upload_object(model_selected), )
        else:
            logger.info(f"chat_id {delete_chat_id} deleted, but it was not the current chat. No action taken on current chat.")
            llm_chat = chat_list.get(current_chat_id, None)
            history = llm_chat.get_history()
            model_nm = llm_chat.get_model_nm()
            system_message = llm_chat.system_message
            return (history, history, user_input, system_message, current_chat_id, set_chat_selector_drop_down(chat_list, current_chat_id),
                    get_file_upload_object(model_nm), )
    except Exception as e:
        # Just in the rare case exception happens. Just reset things so that user can retry.
        err_msg = f"Error loading chat_id: {delete_chat_id}: {str(e)}"
        logger.error(err_msg, exc_info=True)
        return err_response, [], "", "", None, set_chat_selector_drop_down(chat_list, None), gr.File(value="")
        #raise LlmChatException(err_msg) from e

# Launch configuration
# if __name__ == "__main__":
#     print("🚀 Starting Enhanced Multi-LLM Chatbot...")
#     print("📋 Features: Chat Management, File Upload, Clipboard Support")
#