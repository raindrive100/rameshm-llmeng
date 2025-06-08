import mimetypes
import os
import base64
import chardet
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json
from rameshm.llmeng.exception.llm_chat_exception import LlmChatException
from rameshm.llmeng.utils.init_utils import set_environment_logger


def convert_binary_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convert binary file to base64 with metadata.

    Returns:
        Tuple of (base64_string, metadata)
    """
    file_info = get_file_info(file_path)

    if not file_info['exists']:
        return "", {"error": f"File not found: {file_path}", **file_info}

    try:
        with open(file_path, 'rb') as f:
            binary_data = f.read()

        base64_string = base64.b64encode(binary_data).decode('utf-8')

        metadata = {
            "type": "binary",
            "base64_length": len(base64_string),
            "note": "Binary file encoded as base64. May need special handling.",
            "success": True,
            **file_info
        }

        return base64_string, metadata

    except Exception as e:
        return "", {
            "error": f"Failed to encode binary file: {e}",
            **file_info
        }


class FileToLLMConverter:
    """
    Robust file converter that prepares various file types for LLM consumption.
    Handles text files, images, and other binary formats safely.
    """

    # Supported file types
    #TODO: Add more file types as needed. Look to move these to a config file
    TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json',
                       '.csv', '.xml', '.yml', '.yaml', '.sql', '.log', '.ini',
                       '.cfg', '.conf', '.sh', '.bat', '.ps1', '.php', '.rb',
                       '.go', '.rs', '.cpp', '.c', '.h', '.java', '.kt', '.swift'}

    BINARY_EXTENSIONS_SMALL = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.svg',
                         '.pdf', '.docx', '.xlsx', }

    BINARY_EXTENSIONS_LARGE = {'.zip', '.tar', '.gz', '.exe',
                         '.dll', '.so', '.dylib', '.mp3', '.mp4', '.avi', '.mov'}

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


    def file_checks(self, file_path: str) -> (bool,str):
        """
        Perform basic checks on the file.

        Args:
            file_path: Absolute path to the file

        Returns:
            True if file passes checks, False otherwise
        """
        if not os.path.exists(file_path):
            err_msg = f"File not found: {file_path}"
            raise LlmChatException(f"{err_msg}")

        if not os.path.isfile(file_path):
            err_msg = f"Path is not a file: {file_path}"
            raise LlmChatException(f"{err_msg}")

        err_msg = None
        file_info = self.get_file_info(file_path)
        if file_info['size_bytes'] <= 0:
            err_msg = f"File is empty: {file_path}"
        file_extension = file_info['extension']
        file_type_supported = any(file_extension in arr for arr in (self.TEXT_EXTENSIONS, self.BINARY_EXTENSIONS_SMALL, self.BINARY_EXTENSIONS_LARGE))

        print(f"*****DEBUG: File Extension Supported: {file_type_supported}")
        if not file_type_supported:
            err_msg = f"Unsupported file type: {file_info['extension']} for file {file_path}"
        elif file_info['extension'].lower() in self.TEXT_EXTENSIONS and file_info['size_bytes'] > self.max_text_size:
            err_msg = f"Text file too large: {file_info['size_mb']}MB (max: {self.max_text_size / (1024 * 1024)}MB)"
        elif file_info['extension'].lower() in self.BINARY_EXTENSIONS_SMALL and file_info['size_bytes'] > self.max_small_binary_size:
            err_msg = f"Image file too large: {file_info['size_mb']}MB (max: {self.max_small_binary_size / (1024 * 1024)}MB)"
        elif file_info['extension'].lower() in self.BINARY_EXTENSIONS_LARGE and file_info['size_bytes'] > self.max_large_binary_size:
            err_msg = f"Binary file too large: {file_info['size_mb']}MB (max: {self.max_large_binary_size / (1024 * 1024)}MB)"

        if err_msg:
            self.logger.error(err_msg)
            return False, err_msg
        else:
            self.logger.info(f"File checks passed for {file_path}: {file_info}")
            return True, None


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
            self.logger.error(err_msg)
            raise LlmChatException(err_msg) from e
            #return 'utf-8'

    def convert_text_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert text file to LLM-ready format.

        Returns:
            Tuple of (content, metadata)
        """
        file_check_result, err_msg = self.file_checks(file_path)
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


    # TODO: Consider moving the MIME type mapping to a config file or constants module
    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type based on file extension."""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.svg': 'image/svg+xml',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.zip': 'application/zip',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip',
            '.exe': 'application/vnd.microsoft.portable-executable',
            '.dll': 'application/vnd.microsoft.portable-executable',
            '.so': 'application/x-sharedlib',
            '.dylib': 'application/x-dylib',
            '.mp3': 'audio/mpeg',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.xml': 'application/xml',
            '.yml': 'application/x-yaml',
            '.yaml': 'application/x-yaml',
            '.sql': 'application/sql',
            '.log': 'text/plain',
            '.ini': 'text/plain',
            '.cfg': 'text/plain',
            '.conf': 'text/plain',
            '.sh': 'application/x-sh',
            '.bat': 'application/x-msdownload',
            '.ps1': 'application/powershell',
            '.php': 'application/x-httpd-php',
            '.rb': 'application/x-ruby',
            '.go': 'text/x-go',
            '.rs': 'text/x-rust',
            '.cpp': 'text/x-c++src',
            '.c': 'text/x-csrc',
            '.h': 'text/x-chdr',
            '.java': 'text/x-java-source',
            # Add more as needed
        }
        return mime_types.get(extension, "")  # Default to empty string if unknown extension

    def _encode_file_to_base64(self, file_path: str, max_size: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Encode file to base64 string with metadata.
        """
        # Perform file checks
        file_check_result, err_msg = self.file_checks(file_path)
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
                self.max_small_binary_size if file_type in ["image", "binary"] else self.max_text_size
            )
            file_info = self.get_file_info(file_path)
            mime_type = mime_type = mimetypes.guess_type(file_path)[0]
            if not base64_string:
                return "", file_info    # File could be empty or not readable

            # TODO: Need to handle different file types differently
            if file_type == "image":
                mime_type = self._get_mime_type(file_info['extension'])
                result = f"data:{mime_type};base64,{base64_string}"
                metadata['type'] = 'image'
                metadata['mime_type'] = mime_type
            else:
                result = base64_string
                metadata = {"type": "binary", "note": "Binary file...", **file_info}
            return result, metadata
        except Exception as e:
            err_msg = f"Failed to convert file {file_path} to base64: {e}"
            raise LlmChatException(err_msg) from e


    def convert_image_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert image file to base64 format for LLMs.

        Returns:
            Tuple of (base64_string, metadata)
        """
        file_info = self.get_file_info(file_path)

        if file_info['size_bytes'] > self.max_small_binary_size:
            err_msg = f"Image too large: {file_info['size_mb']}MB (max: {self.max_small_binary_size / (1024 * 1024)}MB)"
            self.logger.error(err_msg)
            raise LlmChatException(err_msg)
            # return "", {
            #     "error": f"Image too large: {file_info['size_mb']}MB (max: {self.max_non_text_size / (1024 * 1024)}MB)",
            #     **file_info
            # }

        # MIME type mapping
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.svg': 'image/svg+xml'
        }

        mime_type = mime_types.get(file_info['extension'], 'image/jpeg')

        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()

            base64_string = base64.b64encode(binary_data).decode('utf-8')

            # Create data URL format for LLMs
            data_url = f"data:{mime_type};base64,{base64_string}"

            metadata = {
                "type": "image",
                "mime_type": mime_type,
                "base64_length": len(base64_string),
                "data_url_length": len(data_url),
                "success": True,
                **file_info
            }

            return data_url, metadata

        except Exception as e:
            return "", {
                "error": f"Failed to encode image: {e}",
                **file_info
            }

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

        if file_ext in self.TEXT_EXTENSIONS:
            return self.convert_text_file(file_path)
        elif file_ext in self.BINARY_EXTENSIONS_SMALL or file_ext in self.BINARY_EXTENSIONS_LARGE:
            return self.convert_to_base64(file_path, file_ext)
        else:
            # Unknown extension - try as text first
            err_msg = f"Unsupported file extension: {file_ext} in file {file_path}"
            raise LlmChatException(err_msg)


    def format_for_llm(self, file_path: str, include_metadata: bool = True) -> str:
        """
        Format file content with metadata for LLM consumption.

        Args:
            file_path: Path to the file
            include_metadata: Whether to include metadata in output

        Returns:
            Formatted string ready for LLM
        """
        content, metadata = self.convert_file(file_path)

        if not metadata.get('success', False):
            return f"❌ Error processing file: {metadata.get('error', 'Unknown error')}"

        file_type = metadata.get('type', 'unknown')
        file_name = metadata.get('name', 'unknown')

        if file_type == 'text':
            result = f"📄 **Text File: {file_name}**\n\n"
            if include_metadata:
                result += f"📊 **Metadata:**\n"
                result += f"- Size: {metadata['size_mb']} MB\n"
                result += f"- Encoding: {metadata['encoding']}\n"
                result += f"- Lines: {metadata['line_count']:,}\n"
                result += f"- Characters: {metadata['char_count']:,}\n\n"
            result += f"📝 **Content:**\n```{metadata['extension'][1:]}\n{content}\n```"

        elif file_type == 'image':
            result = f"🖼️ **Image File: {file_name}**\n\n"
            if include_metadata:
                result += f"📊 **Metadata:**\n"
                result += f"- Size: {metadata['size_mb']} MB\n"
                result += f"- MIME Type: {metadata['mime_type']}\n\n"
            result += f"🔗 **Image Data (Base64):**\n{content}"

        elif file_type == 'binary':
            result = f"📦 **Binary File: {file_name}**\n\n"
            if include_metadata:
                result += f"📊 **Metadata:**\n"
                result += f"- Size: {metadata['size_mb']} MB\n"
                result += f"- Base64 Length: {metadata['base64_length']:,} characters\n\n"
            result += f"💾 **Binary Data (Base64):**\n{content}"

        else:
            result = f"❓ **Unknown File: {file_name}**\n\n{content}"

        return result


# Example usage and testing
def main():
    """Example usage of the FileToLLMConverter"""
    converter = FileToLLMConverter()

    # Test with different file types
    test_files = [
        r"c:/temp/test.txt",
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