from typing import List, Dict, Tuple, Any

FILE_DETECTION_CONFIDENCE_LEVEL_NEEDED = 0.6
MAX_COMBINED_SIZE_OF_FILES_UPLOADED = 104*1024*20   # 20MB as max limit for total size of all uploaded files.
MAX_CHAT_NAME_LENGTH=40 # Number of character long for chat name


MODEL_ATTRIBUTES = [
    {"model_nm": "gpt-4o", "model_id": 1, "image_support": True, "raw_pdf_support": False, "text_pdf": True, "supports_file_attachments": True, "supported_types": "Text, Code, PDF, Images", },
    {"model_nm": "gpt-4o-mini", "model_id": 2, "image_support": False, "raw_pdf_support": False, "text_pdf": True, "supports_file_attachments": True, "supported_types": "Text, Code, PDF(Images Ignored)", },
    {"model_nm": "claude-sonnet-4-0", "model_id": 3, "image_support": True, "raw_pdf_support": True, "text_pdf": True,  "supports_file_attachments": True, "supported_types": "Text, Code, PDF, Images", },
    #{"model_nm": "claude-3-5-sonnet-2024102", "model_id": 4, "image_support": True, "raw_pdf_support": True, "text_pdf": True,  "supports_file_attachments": True, "supported_types": "Text, Code, PDF, Images", },
    {"model_nm": "gemini-1.5-flash", "model_id": 5, "image_support": True, "raw_pdf_support": True, "text_pdf": True,  "supports_file_attachments": True, "supported_types": "Text, Code, PDF, Images", },
    {"model_nm": "llama3.2", "model_id": 6, "image_support": False, "raw_pdf_support": False, "text_pdf": True,  "supports_file_attachments": False, "supported_types": "Text, Code, PDF(Images Ignored)", },
    {"model_nm": "llama3.4b", "model_id": 7, "image_support": False, "raw_pdf_support": False, "text_pdf": True,  "supports_file_attachments": False, "supported_types": "Text, Code, PDF(Images Ignored)", },
    ]

# The file extensions mentioned in TEXT_FILE_EXTENSIONS, SUPPORTED_FILE_TYPES are the file types that are supported.
# As we support additional file types add them here.
TEXT_FILE_EXTENSIONS = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json',
                        '.csv', '.xml', '.yml', '.yaml', '.sql', '.log', '.ini',
                        '.cfg', '.conf', '.sh', '.bat', '.ps1', '.php', '.rb',
                        '.go', '.rs', '.cpp', '.c', '.h', '.java', '.kt', '.swift'
                        ]

SUPPORTED_FILE_TYPES = {
    "text": TEXT_FILE_EXTENSIONS,
    "image": ['image', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.svg'],
    "pdf": ['.pdf'],
    "ms-wordx": ['.docx'],
    "ms-excelx": ['.xlsx'],
    "ms-pptx": [".pptx"],
    }

# Define some utility methods.
def get_model_attributes(model_nm: str = None) -> Any:
    if model_nm:
        return [model_attr['supported_types'] for model_attr in MODEL_ATTRIBUTES if model_attr['model_nm'] == model_nm][0]
    else:
        return MODEL_ATTRIBUTES

def get_model_list() -> List[Dict[str, int]]:
    return [{"model_nm": attr['model_nm'], "model_id": attr['model_id']} for attr in MODEL_ATTRIBUTES]

def get_supported_file_types_as_str() -> str:
    """Returns supported file types as string """
    ext_str = ", ".join([", ".join(exts) for exts in SUPPORTED_FILE_TYPES.values()])
    return ext_str

def get_supported_file_types() -> Dict:
    """ Returns SUPPORTED_FILE_TYPES"""
    return SUPPORTED_FILE_TYPES

def get_max_chat_name_length():
    return MAX_CHAT_NAME_LENGTH

def get_file_detection_confidence_needed() -> float:
    return FILE_DETECTION_CONFIDENCE_LEVEL_NEEDED

def get_max_combined_size_of_files_upload() -> int:
    return MAX_COMBINED_SIZE_OF_FILES_UPLOADED

def get_image_handling_models() -> List[str]:
    return [model['model_nm'] for model in MODEL_ATTRIBUTES if model['image_support']]

def get_pdf_image_handling_models() -> List[str]:
    return [model['model_nm'] for model in MODEL_ATTRIBUTES if model['raw_pdf_support']]

def get_models_supporting_file_attachments() -> List[str]:
    return [model['model_nm'] for model in MODEL_ATTRIBUTES if model['supports_file_attachments']]

def get_model_supported_file_types(model_nm):
    """
    Returns the file extensions that the model supports. This is used primarily to set what file types can be uploaded in the
    file_upload section.
    :param model_nm: Name of the Model.
    :return: List of file extensions that the model supports.
    """
    file_extensions_supported = []
    if model_nm in get_models_supporting_file_attachments():
        # File attachments are supported.
        # All models that support file attachments can support Text Files and can support PDF Files without images
        file_extensions_supported += TEXT_FILE_EXTENSIONS
        file_extensions_supported += SUPPORTED_FILE_TYPES['pdf']
        if model_nm in get_image_handling_models():
            # Image handling models can also handle word and excel file so add them as well.
            # TODO: Bundling MS Files with the Image files is a bit hokey. Think a more robust way to handle.
            file_extensions_supported += SUPPORTED_FILE_TYPES['image']
            # TODO: Address later. None of the model APIs are currently supporting Word documents. Hence commenting them out for now.
            file_extensions_supported += SUPPORTED_FILE_TYPES['ms-wordx']
            #file_extensions_supported += SUPPORTED_FILE_TYPES['ms-excelx']
            # file_extensions_supported += SUPPORTED_FILE_TYPES['ms-pptx']
    else:
        # Model doesn't support file attachments.
        pass

    return file_extensions_supported



