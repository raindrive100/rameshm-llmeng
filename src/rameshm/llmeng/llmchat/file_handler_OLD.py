import os
import base64
import chardet
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from rameshm.llmeng.utils.init_utils import set_environment_logger

logger = set_environment_logger()


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


def encode_image_to_base64(file_path: str) -> str:
    """Convert image to base64 string"""
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""


def get_image_info(file_path: str) -> Dict[str, Any]:
    """Get image metadata"""
    try:
        with Image.open(file_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
            }
    except Exception as e:
        return {'error': str(e)}


def detect_encoding_from_sample(file_path, sample_size=5120, confidence=0.8) -> str: # Read 5KB of data
    """Read first 1KB to detect encoding with fallback to utf-8
    file_path: Full absolute Path to the file"""
    try:
        with open(file_path, 'rb') as f:    # Read file in binary mode
            raw_sample = f.read(sample_size)
            if not raw_sample:  # Empty file
                return 'utf-8'

            result = chardet.detect(raw_sample)
            detected_encoding = result.get('encoding')

            # Chardet sometimes returns None or low-confidence results
            if not detected_encoding or result.get('confidence', 0) < confidence:
                return 'utf-8'  # Default fallback

            return detected_encoding
    except Exception:
        err_msg = f"Error detecting encoding for file {file_path}. Defaulting to utf-8."
        logger.error(err_msg)
        return 'utf-8'  # Fallback if detection fails


def process_file_upload(file_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Process uploaded file and extract text/image info
    Returns: (text_content, image_base64, image_format)
    """
    if not file_path:
        return "", None, None

    try:
        encoding = detect_encoding_from_sample(file_path)
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        print(f"Full Path: {file_path} File Name: {file_name}, File Extension: {file_ext} Encoding: {encoding}")

        # TODO: Move the list of Extension to a constants or config file
        if file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv']:
            # Read text files
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # Limit content size to prevent overwhelming the model
                if len(content) > 10000:
                    err_msg = f"File {file_name} is too large to display fully. Showing first 10,000 characters."
                    logger.info(err_msg)
                    content = content[:10000] + "\n... [Content truncated - file too large]"

                return f"File: {file_name}**\n```{file_ext[1:]}\n{content}\n```", None, None
            except Exception as e:
                err_msg = f"Error processing file {file_name}: {e}"
                logger.error(err_msg)
                raise ValueError(f"{err_msg}")

        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            # Handle image files
            image_base64 = encode_image_to_base64(file_path)
            image_info = get_image_info(file_path)

            if image_base64:
                info_text = f"Image uploaded: {file_name}**\n"
                if 'width' in image_info:
                    info_text += f"📐 Size: {image_info['width']}x{image_info['height']} pixels\n"
                    info_text += f"💾 File size: {image_info['size_mb']} MB\n"
                    info_text += f"🎨 Format: {image_info['format']}\n"

                # TODO: Move this to a constants or config file
                # Determine MIME type for API
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }.get(file_ext, 'image/jpeg')

                return info_text, image_base64, mime_type
            else:
                err_msg = f"Failed to encode image {file_name} to base64."
                logger.error(err_msg)
                raise ValueError(f"{err_msg}")
                #return f"❌ Failed to process image: {file_name}", None, None

        elif file_ext == '.pdf':
            err_msg = f"PDF files are not supported for text extraction in this version."
            logger.error()
            raise ValueError(err_msg)
            #return f"📋 **PDF uploaded: {file_name}**\n[Note: PDF text extraction requires additional libraries]", None, None

        else:
            err_msg = f"File Type: {file_ext} is not yet supported"
            raise ValueError(err_msg)
            #return f"📎 **File uploaded: {file_name}** (Type: {file_ext})\n[Unsupported file type for content extraction]", None, None

    except Exception as e:
        err_msg = f"Error processing file {file_path}: {str(e)}"
        logger.error(f"Error processing file {err_msg}")
        raise ValueError(err_msg)  from e
        #return f"❌ Error reading file: {str(e)}", None, None

if __name__ == '__main__':
    test_file_path = r"c:\temp\test.jpg"
    text_file_content = process_file_upload(test_file_path)
    print (f"Text File Content: {text_file_content}")
