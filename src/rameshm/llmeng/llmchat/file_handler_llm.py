import mimetypes
import os
import base64
import chardet
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

from gradio.external import TEXT_FILE_EXTENSIONS

from rameshm.llmeng.exception.llm_chat_exception import LlmChatException
from rameshm.llmeng.utils.init_utils import set_environment_logger


class FileToLLMConverter:
    """
    Robust file converter that prepares various file types for LLM consumption.
    Handles text files, images, and other binary formats safely.
    """

    # Supported file types
    #TODO: Add more file types as needed. Look to move these to a config file
    TEXT_FILE_EXTENSIONS = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json',
                            '.csv', '.xml', '.yml', '.yaml', '.sql', '.log', '.ini',
                            '.cfg', '.conf', '.sh', '.bat', '.ps1', '.php', '.rb',
                            '.go', '.rs', '.cpp', '.c', '.h', '.java', '.kt', '.swift']

    SUPPORTED_FILE_TYPES = {
        "text": TEXT_FILE_EXTENSIONS,
        "image": ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.svg'],
        "binary": ['.pdf']
    }
    #BINARY_EXTENSION = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.svg', # Image
    #                      '.pdf', '.docx', '.xlsx', }
    #
    # BINARY_EXTENSIONS_LARGE = {'.zip', '.tar', '.gz', '.exe',
    #                      '.dll', '.so', '.dylib', '.mp3', '.mp4', '.avi', '.mov'}

    def __init__(self, max_text_size: int = 1_000_000, max_small_binary_size: int = 10_000_000
                 , max_large_binary_size: int = 20_000_000):
        """
        Initialize converter with size limits.

        Args:
            max_text_size: Maximum size for text files (1MB default)
            max_small_binary_size: Maximum size for image and binary files (20MB default)
        """
        self.logger = set_environment_logger()
        self.max_text_size = max_text_size
        self.max_small_binary_size = max_small_binary_size
        self.max_large_binary_size = max_large_binary_size


    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'extension': Path(file_path).suffix.lower(),
                'exists': True
            }
        except Exception as e:
            err_msg = f"Error getting file info for file: {file_path}: {e}"
            self.logger.error(err_msg)
            raise LlmChatException(err_msg) from e
            # return {
            #     'name': os.path.basename(file_path),
            #     'error': str(e),
            #     'exists': False
            # }


    def file_checks(self, file_path: str) -> (bool,str, bool):
        """
        Perform basic checks on the file.

        Args:
            file_path: Absolute path to the file

        Returns:
            True if file passes checks, False otherwise.
            Error message if checks fail, None if checks pass.
            True if it's a fatal error, False otherwise.
        """
        if not os.path.exists(file_path):
            err_msg = f"File not found: {file_path}"
            raise LlmChatException(f"{err_msg}")

        if not os.path.isfile(file_path):
            err_msg = f"Path is not a file: {file_path}"
            raise LlmChatException(f"{err_msg}")

        err_msg = None
        fatal_error = False
        file_info = self.get_file_info(file_path)
        if file_info['size_bytes'] <= 0:
            err_msg = f"File is empty: {file_path}"
        file_extension = file_info['extension']

        file_type_supported = file_extension in [item for sublist in self.SUPPORTED_FILE_TYPES.values() for item in sublist]

        print(f"*****DEBUG: File Extension Is Supported: {file_type_supported}")
        if not file_type_supported:
            err_msg = f"Unsupported file type: {file_info['extension']} for file {file_path}"
            fatal_error = True
        elif file_info['extension'].lower() in self.SUPPORTED_FILE_TYPES['text'] and file_info['size_bytes'] > self.max_text_size:
            err_msg = f"Text file too large: {file_info['size_mb']}MB (max: {self.max_text_size / (1024 * 1024)}MB)"
            fatal_error = True
        elif file_info['extension'].lower() in self.SUPPORTED_FILE_TYPES['binary'] and file_info['size_bytes'] > self.max_small_binary_size:
            err_msg = f"Image file too large: {file_info['size_mb']}MB (max: {self.max_small_binary_size / (1024 * 1024)}MB)"
            fatal_error = True

        if not err_msg and not fatal_error:
            self.logger.info(f"File checks passed for {file_path}: {file_info}")
            return True, None, False
        else:
            # For now raising exception for any file check errors. Later we can handle this more gracefully
            raise LlmChatException(err_msg)


    def detect_encoding(self, file_path: str, sample_size: int = 8192) -> str:
        """
        Robustly detect file detected_encoding with multiple fallbacks.

        Args:
            file_path: Absolute Path to the file
            sample_size: Bytes to sample for detection (default 8KB)

        Returns:
            Detected detected_encoding string
        """
        try:
            with open(file_path, 'rb') as f: # Read file in binary mode
                raw_sample = f.read(sample_size)

            if not raw_sample:
                return 'utf-8'

            # Try chardet detection
            result = chardet.detect(raw_sample)
            detected_encoding = result.get('detected_encoding')
            confidence = result.get('confidence', 0)

            encodings_to_try = []

            # Helper function to avoid case-insensitive duplicates
            def add_encoding_if_new(encoding):
                if encoding and encoding.lower() not in [e.lower() for e in encodings_to_try]:
                    encodings_to_try.append(encoding)

            # 1. High confidence chardet result
            if detected_encoding and confidence >= 0.8:
                add_encoding_if_new(detected_encoding)

            # 2. Medium confidence for common encodings only
            elif (detected_encoding and confidence >= 0.6 and
                  detected_encoding.lower() in ['utf-8', 'ascii', 'latin-1', 'windows-1252']):
                add_encoding_if_new(detected_encoding)

            # 3. Standard fallbacks
            standard_encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'windows-1252', 'ascii']
            for enc in standard_encodings:
                add_encoding_if_new(enc)

            # 4. Try each encoding
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1024)
                    return encoding
                except (UnicodeDecodeError, LookupError):
                    continue

            # TODO: Consider if we should use 'utf-8' or 'latin-1' as the final fallback or raise exception?
            err_msg = f"Unknown Encoding: {encoding} encountered with low confidence for {file_path}. Defaulting to 'latin-1'."
            self.logger.error(err_msg)
            raise LlmChatException(err_msg)
            #return 'latin-1'

        except Exception as e:
            # TODO: Consider if we should use 'utf-8' or 'latin-1' as the final fallback or raise exception?
            err_msg = f"Encoding detection error for file {file_path}: {e}."
            raise LlmChatException(err_msg) from e
            #return 'utf-8'

    def convert_text_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert text file to LLM-ready format.

        Returns:
            Tuple of (content, metadata)
        """
        file_check_result, err_msg, fatal_error = self.file_checks(file_path)
        if not file_check_result:
            raise LlmChatException(err_msg)

        file_info = self.get_file_info(file_path)
        encoding = self.detect_encoding(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]

        try:
            with open(file_path, 'r', encoding=encoding) as f:  # Opening file with detected encoding, hence only "r" and not "rb"
                content = f.read()

            metadata = {
                "type": "text",
                'mime_type': mime_type,
                "encoding": encoding,
                "content_length": len(content),
                #"char_count": len(content),
                "line_count": content.count('\n') + 1,
                "success": True,
                **file_info
            }

            return content, metadata

        except Exception as e:
            err_msg = f"Failed to read file {file_path} with encoding {encoding}: {e}"
            raise LlmChatException(err_msg) from e
            # return "", {
            #     "error": f"Failed to read file: {e}",
            #     "encoding_attempted": encoding,
            #     **file_info
            # }


    def _encode_file_to_base64(self, file_path: str, max_size: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Encode file to base64 string with metadata.
        """
        # Perform file checks
        file_check_result, err_msg, fatal_error = self.file_checks(file_path)
        if not file_check_result:
            raise LlmChatException(err_msg)
        file_info = self.get_file_info(file_path)

        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            base64_string = base64.b64encode(binary_data).decode('utf-8')

            metadata = {
                "content_length": len(base64_string),
                "base64_length": len(base64_string),
                "success": True,
                **file_info
            }

            return base64_string, metadata

        except Exception as e:
            err_msg = f"Failed to encode file {file_path} to base64: {e}"
            raise LlmChatException(err_msg) from e

    def convert_to_base64(self, file_path: str, file_type: str) -> Tuple[str, Dict[str, Any]]:
        """Universal base64 converter"""
        # Common encoding logic
        try:
            base64_string, metadata = self._encode_file_to_base64(
                file_path,
                self.max_small_binary_size
            )

            mime_type = mimetypes.guess_type(file_path)[0]
            metadata['type'] = file_type
            metadata['mime_type'] = mime_type

            return base64_string, metadata
            # file_info = self.get_file_info(file_path)
            # mime_type = mimetypes.guess_type(file_path)[0]
            # if not base64_string:
            #     return "", file_info    # File could be empty or not readable
            #
            # # TODO: Need to handle different file types differently
            # return result, metadata
            # if file_type == "image":
            #     result = f"data:{mime_type};base64,{base64_string}"
            #     metadata['type'] = 'image'
            #     metadata['mime_type'] = mime_type
            # else:
            #     result = base64_string
            #     metadata = {"type": "binary", "mime_type": mime_type, "note": "Binary file...", **file_info}
            # return result, metadata
        except Exception as e:
            err_msg = f"Failed to convert file {file_path} to base64: {e}"
            raise LlmChatException(err_msg) from e


    def convert_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main method to convert any file to LLM-ready format.

        Args:
            file_path: Path to the file to convert

        Returns:
            Tuple of (converted_content, metadata)
        """
        if not os.path.exists(file_path):
            err_msg = f"File not found: {file_path}"
            raise LlmChatException(err_msg)

        file_ext = Path(file_path).suffix.lower()

        if file_ext in self.SUPPORTED_FILE_TYPES['text']:
            return self.convert_text_file(file_path)
        elif file_ext in self.SUPPORTED_FILE_TYPES['image']:
            return self.convert_to_base64(file_path, 'image')
        elif file_ext in self.SUPPORTED_FILE_TYPES['binary']:
            return self.convert_to_base64(file_path, 'binary')
        else:
            err_msg = f"Unsupported file extension: {file_ext} in file {file_path}"
            raise LlmChatException(err_msg)


    def format_for_llm(self, file_path: str, include_metadata: bool = True) -> tuple[str, Dict[str, str]]:
        """
        Format file content with metadata for LLM consumption.

        Args:
            file_path: Path to the file
            include_metadata: Whether to include metadata in output

        Returns:
            Formatted string ready for LLM
        """
        try:
            content, metadata = self.convert_file(file_path)

            file_type = metadata.get('type', 'unknown')
            file_name = metadata.get('name', 'unknown')
            mime_type = metadata.get('mime_type', 'unknown')
            if file_type == 'text':
                # For text files, we can include the content directly
                return content, metadata
            elif file_type in ('image', 'binary'):
                result = f"data:{mime_type};base64,{content}"
                return result, metadata
            else:
                raise LlmChatException(f"Unsupported file type: {file_type} for file {file_name}")
        except Exception as e:
            err_msg = f"Error formatting file {file_path} for LLM: {e}"
            raise LlmChatException(err_msg) from e


# Example usage and testing
def main():
    """Example usage of the FileToLLMConverter"""
    converter = FileToLLMConverter()

    # Test with different file types
    test_files = [
        #r"c:/temp/test.txt",
        r"c:\temp\test.jpg"
        #"data.csv",
        #"script.py"
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'=' * 50}")
            print(f"Processing: {file_path}")
            print('=' * 50)

            # Get formatted output for LLM
            llm_ready_content = converter.format_for_llm(file_path)
            print(llm_ready_content)

            # Or get raw content and metadata separately
            content, metadata = converter.convert_file(file_path)
            print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
        else:
            print(f"⚠️  File not found: {file_path}")


if __name__ == "__main__":
    main()