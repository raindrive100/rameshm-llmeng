import base64
import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import chardet
import puremagic
import pymupdf
from PIL import Image
from docx import Document

import rameshm.llmeng.llmchat.chat_constants as chat_constants
from rameshm.llmeng.exception.llm_chat_exception import LlmChatException
from rameshm.llmeng.utils.init_utils import set_environment_logger


class FileToLLMConverter:
    """
    Robust file converter that prepares various file types for LLM consumption.
    Handles text files, images, and other binary formats safely.
    """

    def __init__(
        self, 
        max_text_size: int = 1_000_000,
        max_small_binary_size: int = 10_000_000,
        max_large_binary_size: int = 20_000_000
    ) -> None:
        """
        Initialize file converter with size limits.

        Args:
            max_text_size: Maximum size for text files (1MB default)
            max_small_binary_size: Maximum size for image and binary files (10MB default)
            max_large_binary_size: Maximum size for large binary files (20MB default)
        """
        self.logger = set_environment_logger()
        self.max_text_size = max_text_size
        self.max_small_binary_size = max_small_binary_size
        self.max_large_binary_size = max_large_binary_size


    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        Args:
            file_path: Absolute path to the file
        Returns:
            Dictionary containing file information
        """
        self.logger.debug(f"Getting file info: {file_path}")
        size_bytes = os.path.getsize(file_path)
        return {
            'name': os.path.basename(file_path),
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'extension': Path(file_path).suffix.lower(),
            'exists': True
        }


    def detect_non_text_file_type(self, file_path: str) -> Tuple[Optional[str], float]:
        """
        Detect MIME type and confidence for non-text files using puremagic.
        Args:
            file_path: Absolute path to the file
        Returns:
            Tuple of (mime_type, confidence)
        """
        self.logger.debug(f"Detect non-text file type: {file_path}")
        magic_infos = puremagic.magic_file(file_path)
        max_confidence = 0
        mime_type = None

        for magic_info in magic_infos:
            if magic_info.confidence > max_confidence:
                max_confidence = magic_info.confidence
                mime_type = magic_info.mime_type

        self.logger.debug(f"Detected MIME type: {mime_type} with confidence: {max_confidence}")
        return mime_type, max_confidence


    def detect_text_file_encoding(self, file_path: str, sample_size: int = 8192) -> Tuple[str, float]:
        """
        Detect file encoding using chardet.
        Args:
            file_path: Absolute path to the file
            sample_size: Bytes to sample for detection (default 8KB)
        Returns:
            Tuple of (encoding, confidence)
        """
        self.logger.debug(f"Detect text file encoding: {file_path}")
        with open(file_path, 'rb') as f:
            raw_sample = f.read(sample_size)

        if not raw_sample:
            return None, 0.0

        result = chardet.detect(raw_sample)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)

        self.logger.debug(f"Detected text encoding: {encoding} with confidence: {confidence}")
        return encoding, confidence


    def is_valid_file_type(self, file_path: str, confidence_threshold: float = 1.0) -> bool:
        """
        Validate file type based on extension and content confidence.
        Args:
            file_path: Absolute path to the file
            confidence_threshold: Minimum confidence level required
        Returns:
            True if file type is valid, False otherwise
        """
        self.logger.debug(f"Validate file type: {file_path}")
        file_extension = Path(file_path).suffix.lower()

        # Check if file type is supported
        if not chat_constants.get_file_type_from_extension(file_extension):
            self.logger.warning(f"Unsupported file type: {file_extension} for {file_path}")
            return False

        # Get MIME types and confidence levels
        mime_type = mimetypes.guess_type(file_path)[0]
        mime_type_binary, confidence_binary = self.detect_non_text_file_type(file_path)
        encoding, confidence_text = self.detect_text_file_encoding(file_path)
        confidence = max(confidence_binary, confidence_text)
        self.logger.debug(f"Identified Types: text encoding: {encoding} with confidence: {confidence} "
                          f"and Binary Mime Type: {mime_type_binary} and Binary Confidence: {confidence_binary}")

        # Determine the primary content type based on detection confidence
        if confidence_text >= confidence_binary and encoding:  # Prioritize text if confidence is higher or equal
            detected_content_type = "text"
            actual_mime_type = f"text/{encoding.lower()}"  # More specific text mime type
            is_valid = confidence_text >= confidence_threshold
            self.logger.debug(
                f"Content-based text validation: encoding={encoding}, confidence={confidence_text}, threshold={confidence_threshold}")
        elif mime_type_binary:  # Fallback to binary detection
            detected_content_type = "binary"
            actual_mime_type = mime_type_binary
            is_valid = confidence_binary >= confidence_threshold
            self.logger.debug(
                f"Content-based binary validation: mime={mime_type_binary}, confidence={confidence_binary}, threshold={confidence_threshold}")
        else:
            is_valid = False  # No confident detection
            self.logger.warning(f"No confident content type detected for {file_path}")

        # Optionally, add a check for consistency with extension-based guess. This should never unless the content is different from extension type
        if is_valid and mime_type and not actual_mime_type.startswith(mime_type.split('/')[0]):
            self.logger.warning(f"Content type ({actual_mime_type}) inconsistent with extension guess ({mime_type}) for {file_path}")
            is_valid = False

        if not is_valid:
            self.logger.warning(f"File validation failed for {file_path}: confidence={confidence} < threshold={confidence_threshold}")

        return is_valid

    def is_valid_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Perform comprehensive file validation.
        Args:
            file_path: Absolute path to the file
        Returns:
            Tuple of (is_valid, error_message)
        """
        self.logger.debug(f"Validate File: {file_path}")
        file_info = self.get_file_info(file_path)

        # Basic file existence and type checks
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"

        if file_info['size_bytes'] <= 0:
            return False, f"File is empty: {file_path}"

        # File type validation
        if not chat_constants.get_file_type_from_extension(file_info['extension'].lower()):
            return False, f"Unsupported file type: {file_info['extension']}"

        # Content validation
        if not self.is_valid_file_type(file_path, chat_constants.get_file_detection_confidence_needed()):
            return False, f"Invalid file content for: {file_path}"

        self.logger.info(f"File validation passed for {file_path}: {file_info}")
        return True, None

    def _create_common_metadata(self, file_path: str, file_type: str, content: str,
                                extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        file_info = self.get_file_info(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]
        metadata = {
            'file_type': file_type,
            'mime_type': mime_type,
            'content_length': len(content),
            **file_info
        }
        if extra_info:
            metadata.update(extra_info)
        return metadata


    def extract_text_from_pdf_file(self, file_path: str, include_images: bool = False, file_type: str = "pdf") -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF file.
        Args:
            file_path: Absolute path to the PDF file
            include_images: If True, returns base64 encoded content including images
            file_type: "pdf" is passed.
        Returns:
            Tuple of (content, metadata)
        Raises:
            LlmChatException: If PDF extraction fails
        """
        self.logger.debug(f"Extract text from PDF for File: {file_path}")
        file_info = self.get_file_info(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]

        if include_images:
            with open(file_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
                content_type = 'base64'
        else:
            with pymupdf.open(file_path) as doc:
                content = "\n--- Page Break ---\n".join([page.get_text() for page in doc])
                content_type = 'text'

        metadata = self._create_common_metadata(file_path, file_type, content, {'content_type': content_type})

        self.logger.debug(f"Successfully extracted PDF content from {file_path}")
        return content, metadata


    def convert_docx_to_text(self, file_path: str, file_type: str = "ms-wordx") -> Tuple[str, Dict[str, Any]]:
        """
        Convert DOCX file to text.
        Args:
            file_path: Absolute path to the DOCX file
            file_type: ms-wordx is typically passed.
        Returns:
            Tuple of (text_content, metadata)
        Raises:
            LlmChatException: If DOCX conversion fails
        """
        self.logger.debug(f"Convert DOCX to text for File: {file_path}")
        document = Document(file_path)
        text_content = "\n".join(paragraph.text for paragraph in document.paragraphs)

        metadata = self._create_common_metadata(file_path, file_type, text_content)

        self.logger.debug(f"Successfully converted DOCX to text: {file_path}")
        return text_content, metadata


    def convert_text_file_to_text(self, file_path: str, file_type: str = "text") -> Tuple[str, Dict[str, Any]]:
        """
        Convert text file to LLM-ready format.
        Args:
            file_path: Absolute path to the text file
            file_type: "text" is typically passed
        Returns:
            Tuple of (content, metadata)
        Raises:
            LlmChatException: If text file conversion fails
        """
        self.logger.debug(f"Convert text file to text for File: {file_path}")
        file_info = self.get_file_info(file_path)
        encoding, confidence = self.detect_text_file_encoding(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]

        # Use UTF-8 for ASCII encoding to ensure compatibility
        encoding = "utf-8" if "ascii" in encoding else encoding

        with open(file_path, 'r', encoding=encoding) as f: # Not "rb" because encoding is specified.
            content = f.read()
        metadata = self._create_common_metadata(file_path, file_type, content,
                                                {'encoding': encoding, 'line_count': content.count('\n') + 1,
                                                 'success': True})

        self.logger.debug(f"Successfully converted text file {file_path}")
        return content, metadata


    def convert_image_to_base64(self, file_path: str, file_type: str = "image") -> Tuple[str, Dict[str, Any]]:
        """
        Convert image file to base64 string.
        Args:
            file_path: Absolute path to the image file
            file_type: "image" is typically passed.
        Returns:
            Tuple of (base64_string, metadata)
        Raises:
            LlmChatException: If image conversion fails
        """
        self.logger.debug(f"Convert image file to base64 string for File: {file_path}")

        # Convert image to RGB and save as JPEG to standardize the format and
        # ensure compatibility with models. This is a lossy conversion.
        image = Image.open(file_path).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        metadata = self._create_common_metadata(file_path, file_type, base64_string)

        self.logger.debug(f"Successfully converted image to base64: {file_path}")
        return base64_string, metadata


    def get_file_type(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine file type based on extension.
        Args:
            file_path: Absolute path to the file
        Returns:
            Tuple of (file_type, mime_type)
        """
        mime_type = mimetypes.guess_type(file_path)[0]
        file_ext = Path(file_path).suffix.lower()
        file_type = chat_constants.get_file_type_from_extension(file_ext)

        if not file_type:
            err_msg = f"Unsupported file type: {file_ext} for file: {file_path}"
            self.logger.warning(err_msg)
            return None, mime_type

        return file_type, mime_type

    def convert_file_to_str(self, file_path: str, check_file_validity: bool = False,
                            include_images_in_pdf = False) -> Tuple[str, Dict[str, Any]]:
        """
        TODO: Add code to support MS Word/Excel and Audio files.
        Main method to convert any file to LLM-ready format.
        Args:
            file_path: Path to the file to convert
            check_file_validity: True -> Series of validity checks to be performed; False -> No validity checks performed (probably aleady done by caller)
            include_images_in_pdf: "True" to include images contained within PDF file (larger output) "False" to exclude images in PDF file
        Returns:
            Tuple of (converted_content, metadata)
        """
        self.logger.debug(f"Starting converting file: {file_path} to text/image/pdf string")
        if check_file_validity:
            file_passed_checks, err_msg_file_checks = self.is_valid_file(file_path)
        else:
            # The caller would have checked validity already.
            file_passed_checks, err_msg_file_checks = True, ""
        self.logger.debug(f"File checks Passed?: {file_passed_checks}")
        if file_passed_checks:
            file_ext = Path(file_path).suffix.lower()
            file_type, mime_type = self.get_file_type(file_path)
            if file_type == "text":
                return self.convert_text_file_to_text(file_path, file_type)
            elif file_type == "image":
                return self.convert_image_to_base64(file_path, file_type)
            elif file_type == "pdf":
                return self.extract_text_from_pdf_file(file_path, include_images_in_pdf, file_type)
            elif file_type in ("ms-wordx"):
                return self.convert_docx_to_text(file_path, file_type)
            else:
                err_msg = f"Unsupported file extension: {file_ext} in file {file_path}"
                raise LlmChatException(err_msg)
        else:
            err_msg = f"File checks failed for file: {file_path}. Error: {err_msg_file_checks}"
            self.logger.warning(err_msg)
            raise LlmChatException(err_msg)


# Example usage and testing
def main():
    """Example usage of the FileToLLMConverter"""
    converter = FileToLLMConverter()

    # Test with different file types
    test_files = [
        r"c:/temp/test.txt",
        r"c:\temp\test.jpg",
        r"c:\temp\test_1.pdf"
        #"data.csv",
        #"script.py"
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'=' * 50}")
            print(f"Processing: {file_path}")
            print('=' * 50)

            # Get formatted output for LLM
            #llm_ready_content = converter.format_for_llm(file_path)
            #print(llm_ready_content)

            # Or get raw content and metadata separately
            content, metadata = converter.convert_file_to_str(file_path, include_images_in_pdf=True)
            print(f"Content: {content} \nMetadata: {json.dumps(metadata, indent=2)}")
        else:
            print(f"⚠️  File not found: {file_path}")


if __name__ == "__main__":
    main()