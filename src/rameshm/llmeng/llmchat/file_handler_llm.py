import mimetypes # To determine mime-type of the file
import puremagic # To determine the file type, i.e. image vs. pdf vs. text
import os
from PIL import Image # for processing images
import io
import base64 # to convert image files to base64
import pymupdf # to convert pdf files to text
from docx import Document
import chardet # used for detecting text file encoding
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

from traits.trait_types import false

from rameshm.llmeng.exception.llm_chat_exception import LlmChatException
from rameshm.llmeng.utils.init_utils import set_environment_logger
import rameshm.llmeng.llmchat.chat_constants as chat_constants

class FileToLLMConverter:
    """
    Robust file converter that prepares various file types for LLM consumption.
    Handles text files, images, and other binary formats safely.
    """

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
            #stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'size_bytes': os.path.getsize(file_path), #  stat.st_size,
                'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
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


    def detect_non_text_file_type(self, file_path: str) -> Tuple[Optional[str], float]:
        max_confidence = 0
        mime_type = None
        self.logger.debug(f"Detecting mime_type and confidence levels for non-text files for file: {file_path}")
        magic_infos = puremagic.magic_file(file_path) # returns a list containing mime-types and confidence
        for magic_info in magic_infos:
            if magic_info.confidence > max_confidence:
                max_confidence = magic_info.confidence
                mime_type = magic_info.mime_type
        self.logger.debug(f"mime_type: {mime_type} Confidence: {max_confidence} for file: {file_path}")
        return mime_type, max_confidence

    def detect_text_file_encoding(self, file_path: str, sample_size: int = 8192) -> Tuple[str, float]:
        """
        Detect file detected_encoding. Works best for text files

        Args:
            file_path: Absolute Path to the file
            sample_size: Bytes to sample for detection (default 8KB)

        Returns:
            Detected detected_encoding string
        """
        self.logger.debug(f"Detecting Text encoding for file: {file_path}")
        try:
            with open(file_path, 'rb') as f: # Read file in binary mode
                raw_sample = f.read(sample_size)

            if not raw_sample:
                return None, 0  # Unable to read the file.

            # Try chardet detection
            result = chardet.detect(raw_sample)
            detected_encoding = result.get('encoding')
            confidence = result.get('confidence', 0)
            self.logger.debug(f"Finished encoding for Text with encoding: {detected_encoding} with confidence: {confidence} for file: {file_path}")
            return detected_encoding, confidence
        except Exception as e:
            err_msg = f"Encoding detection error for file {file_path}: {e}."
            self.logger.error(err_msg, exc_info=True)
            raise LlmChatException(err_msg) from e


    def is_valid_file_type(self, file_path: str, confidence_level_needed = 1.0) -> bool:
        """ Make sure that file extensions matches and the file contents match with high confidence.
        Return True if it does else False
        """
        self.logger.debug(f"Starting validation of file type for file: {file_path} with "
                          f"desired confidence of: {confidence_level_needed}")
        is_valid_type = False  # Initialized to false and then set to pass if all checks pass.

        file_extension = Path(file_path).suffix.lower()
        file_type_supported = file_extension.lower() in chat_constants.get_supported_file_types_as_str()

        mime_type = mimetypes.guess_type(file_path)[0]
        self.logger.debug(f"Mime type is {mime_type} for file: {file_path}")
        mime_type_binary, confidence_binary = self.detect_non_text_file_type(file_path)
        encoding, confidence_text = self.detect_text_file_encoding(file_path)
        confidence = confidence_binary if confidence_binary > confidence_text else confidence_text

        # Make sure that the mime_types match the confidence from their corresponding confidence functions.
        # This ensures that the file extensions are
        # properly matching the type of data stored in the files. For example, a text file stored as .jpg will be rejected.
        if not file_type_supported:
            is_valid_type = False
            msg = f"File type: {file_extension} is not supported for file: {file_path}"
            self.logger.warning(msg)
        elif "text" in mime_type and confidence_text >= confidence_level_needed:
            is_valid_type = True
            msg = (
                f"Confidence level of {confidence_text} in identifying the file is >= {confidence_level_needed} MimeType: {mime_type}. "
                f"Encoding: {encoding}. Returning True for is_valid_file_type")
            self.logger.debug(msg)
        elif mime_type == mime_type_binary and confidence_binary >= confidence_level_needed:
            # Treating everything other than "text" file as a Binary File. We may need to revist this.
            is_valid_type = True
            msg = (f"Confidence level of {confidence_binary} in identifying the file is >= {confidence_level_needed} MimeType: {mime_type}. "
                   f"Returning True for is_valid_file_type")
            self.logger.debug(msg)
        else:
            # Confidence levels are lower than required.
            is_valid_type = False
            msg = (f"Confidence_binary: {confidence_binary}, confidence_text: {confidence_text} are both lower than {confidence_level_needed}."
                   f"Make sure that the file extensions match the file contents")
            self.logger.warning(msg)

        self.logger.debug(f"Completed validation of file type for file: {file_path} with "
                          f"desired confidence of: {confidence_level_needed}")

        return is_valid_type

    def is_valid_file(self, file_path: str) -> (bool, str):
        """
        Perform basic checks on the file.

        Args:
            file_path: Absolute path to the file

        Returns:
            True if file passes checks, False otherwise.
            Error message if checks fail, None if checks pass.
        """
        self.logger.debug(f"Performing file_checks on file: {file_path}")
        file_info = self.get_file_info(file_path)
        err_msg = None

        # Perform multiple checks on the file.
        if not os.path.exists(file_path):
            err_msg = f"File not found: {file_path}"
        elif not os.path.isfile(file_path):
            err_msg = f"Path is not a file: {file_path}"
        elif not file_info['size_bytes'] > 0:
            err_msg = f"File is empty: {file_path}"
        elif not file_info['extension'].lower() in chat_constants.get_supported_file_types_as_str():
            err_msg = f"{file_info['extension']} is not a supported file type"
        elif not self.is_valid_file_type(file_path, chat_constants.get_file_detection_confidence_needed()):
            err_msg = f"Unsupported file type: {file_info['extension']} for file {file_path}"

        if not err_msg:
            self.logger.info(f"File checks passed for {file_path}: {file_info}")
            return True, None
        else:
            # For now raising exception for any file check errors. Later we can handle this more gracefully
            self.logger.warning(f"File checks Failed for {file_path}. Error: {err_msg}: File Info: {file_info}")
            raise LlmChatException(err_msg)


    def extract_text_from_pdf_file(self, file_path: str, include_images=False,
                                   file_type: str= "pdf") -> Tuple[str, Dict[str, Any]]:
        """
        Extracts text from PDF.
        :param file_path: Absolute file location
        :param include_images: Set it to True if the model can handle imagaes etc. within PDF.
                                If "True" returns base64
                                If "False: returns plan text. Images etc from PDF are dropped.
        :return:
        """
        # PDF extraction
        self.logger.debug(f"Starting converting PDF File to string: {file_path} with include_images: {include_images}")
        print(f"File type is:{type(file_path)} Extracting text from PDF File: {file_path}")
        base64_string = ""
        text_string = ""
        try:
            if include_images:
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                    base64_string = base64.b64encode(binary_data).decode('utf-8')
            else:
                with pymupdf.open(file_path, filetype="pdf") as doc:  # open document
                    text_string = chr(12).join([page.get_text() for page in doc])

            file_info = self.get_file_info(file_path)
            mime_type = mimetypes.guess_type(file_path)[0]
            metadata = {
                'file_type': file_type,
                'mime_type': mime_type,
                'non_base64_length': len(text_string),
                'base64_length': len(base64_string),
                **file_info
            }
            self.logger.debug(f"Finished converting PDF File to string mime_type: {mime_type} File: {file_path}")
            return ((base64_string, metadata) if include_images else (text_string, metadata))
        except Exception as e:
            err_msg = f"Failed to read file {file_path} Error: {e}"
            self.logger.error(err_msg, exc_info=True)
            raise LlmChatException(err_msg) from e


    def docx_to_text(self, docx_path, file_type:str="ms-wordx")-> List[Dict[str, str]]:
        try:
            document = Document(docx_path)
            text_content = []
            full_text = ""
            for paragraph in document.paragraphs:
                text_content.append(paragraph.text)
            full_text = "\n".join(text_content)

            # Build Metadata into
            file_info = self.get_file_info(docx_path)
            mime_type = mimetypes.guess_type(docx_path)[0]
            metadata = {
                'file_type': file_type,
                'mime_type': mime_type,
                'non_base64_length': len(full_text),
                'base64_length': 0,
                **file_info
            }
            self.logger.debug(f"Finished converting Microsoft Word to Plain Text File: {docx_path}")
            return full_text, metadata
        except Exception as e:
            err_msg = f"Failed to read file {docx_path} Error: {e}"
            self.logger.error(err_msg, exc_info=True)
            raise LlmChatException(err_msg) from e


    # def extract_text_from_ms_file(self, file_path: str, file_type: str) -> Tuple[str, Dict[str, Any]]:
    #     """
    #     Extracts text from Microsoft Word and Excel Files.
    #     :param file_path: Absolute file location
    #     :param file_type: ms-excel or ms-word or ms-powerpoint
    #     :return:
    #     """
    #     # Microsoft File extraction
    #     self.logger.debug(f"Starting converting Microsoft File to string: {file_path}")
    #     print(f"File type is:{type(file_path)} Extracting text from MS File: {file_path}")
    #     base64_string = ""
    #     try:
    #         with open(file_path, 'rb') as f:
    #             binary_data = f.read()
    #             base64_string = base64.b64encode(binary_data).decode('utf-8')
    #
    #         file_info = self.get_file_info(file_path)
    #         mime_type = mimetypes.guess_type(file_path)[0]
    #         metadata = {
    #             'file_type': file_type,
    #             'mime_type': mime_type,
    #             'non_base64_length': 0,
    #             'base64_length': len(base64_string),
    #             **file_info
    #         }
    #         self.logger.debug(f"Finished converting Microsoft File to string mime_type: {mime_type} File: {file_path}")
    #         return base64_string, metadata
    #     except Exception as e:
    #         err_msg = f"Failed to read file {file_path} Error: {e}"
    #         self.logger.error(err_msg, exc_info=True)
    #         raise LlmChatException(err_msg) from e


    def extract_text_from_text_file(self, file_path: str, file_type: str = "text") -> Tuple[str, Dict[str, Any]]:
        """
        Convert text file to LLM-ready format.

        Returns:
            Tuple of (content, metadata)
        """
        self.logger.debug(f"Starting converting text file to string: {file_path}")

        file_info = self.get_file_info(file_path)
        encoding, confidence = self.detect_text_file_encoding(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]
        encoding = "utf-8" if "ascii" in encoding else encoding # If Ascii encoding then use utf-8 to be safe.
        self.logger.debug(f"Using encoding: {encoding} with confidence: {confidence} and mime_type: {mime_type}")
        try:
            with open(file_path, 'r', encoding=encoding) as f:  # Opening file with detected encoding, hence only "r" and not "rb"
                content = f.read()

            metadata = {
                "file_type": file_type,
                'mime_type': mime_type,
                "encoding": encoding,
                "non_base64_length": len(content),
                "base64_length": 0,
                #"char_count": len(content),
                "line_count": content.count('\n') + 1,
                "success": True,
                **file_info
            }
            self.logger.debug(f"Finished converting text file to string. mime_type: {mime_type} File: {file_path}")
            return content, metadata

        except Exception as e:
            err_msg = f"Failed to read file {file_path} with encoding {encoding}: {e}"
            self.logger.error(err_msg, exc_info=True)
            raise LlmChatException(err_msg) from e
            # return "", {
            #     "error": f"Failed to read file: {e}",
            #     "encoding_attempted": encoding,
            #     **file_info
            # }


    # def _encode_file_to_base64(self, file_path: str, max_size: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    #     """Encode file to base64 string with metadata.
    #     """
    #     self.logger.debug(f"Start encoding file: {file_path} to base64")
    #
    #     file_info = self.get_file_info(file_path)
    #
    #     try:
    #         with open(file_path, 'rb') as f:
    #             binary_data = f.read()
    #         base64_string = base64.b64encode(binary_data).decode('utf-8')
    #
    #         metadata = {
    #             "content_length": len(base64_string),
    #             "base64_length": len(base64_string),
    #             "success": True,
    #             **file_info
    #         }
    #         self.logger.debug(f"Finished encoding file: {file_path} to base64")
    #         return base64_string, metadata
    #     except Exception as e:
    #         err_msg = f"Failed to encode file {file_path} to base64: {e}"
    #         raise LlmChatException(err_msg) from e


    def extract_base64_from_image_file(self, file_path: str,
                                       file_type: str = "image") -> Tuple[str, Dict[str, Any]]:
        """Universal base64 converter"""
        self.logger.debug(f"Starting converting Image file: {file_path} to base64")
        try:
            image = Image.open(file_path).convert("RGB") # Some LLMs work well with RGB mode
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG") # Converting to uniform format.
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Below is another way to get in base64 but may not work with some of the LLMs hence using the above
            # even though the above is bit slower.
            # with open(file_path, 'rb') as f:
            #     binary_data = f.read()
            #     base64_string = base64.b64encode(binary_data).decode('utf-8')

            file_info = self.get_file_info(file_path)
            mime_type = mimetypes.guess_type(file_path)[0]
            metadata = {
                'file_type': file_type,
                'mime_type': mime_type,
                'non_base64_length': 0,
                'base64_length': len(base64_string),
                **file_info
            }
            self.logger.debug(f"Finished converting Image file to base64. mime_type: {mime_type} File: {file_path}")
            return base64_string, metadata
        except Exception as e:
            err_msg = f"Failed to convert file {file_path} to base64: {e}"
            self.logger.error(err_msg)
            raise LlmChatException(err_msg) from e


    def get_file_type(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns whether a file is text or pdf or image file. Modify this code as add Additional file types like Audio, PPT etc..
        :param file_path: path can be absolute path or just file name
        :return: Tuple of file type (text, pdf, image) and mime-type
        """
        mime_type = mimetypes.guess_type(file_path)[0]
        file_ext = Path(file_path).suffix.lower()
        file_type = None
        for k, v in chat_constants.get_supported_file_types().items():
            if file_ext in chat_constants.get_supported_file_types()[k]:
                file_type = k

        if file_type:
            return file_type, mime_type
        else:
            err_msg = f"Unsupported file type: {file_ext} for file: {file_path}"
            self.logger.warning(err_msg)
            return None, mime_type


    def convert_file_to_str(self, file_path: str, check_file_validity: bool = false,
                            include_images_in_pdf = False) -> Tuple[str, Dict[str, Any]]:
        """
        Main method to convert any file to LLM-ready format.
        TODO: Add code to support MS Word/Excel and Audio files.

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
        if file_passed_checks:
            file_ext = Path(file_path).suffix.lower()
            file_type, mime_type = self.get_file_type(file_path)
            if file_type == "text":
                return self.extract_text_from_text_file(file_path, file_type)
            elif file_type == "image":
                return self.extract_base64_from_image_file(file_path, file_type)
            elif file_type == "pdf":
                return self.extract_text_from_pdf_file(file_path, include_images_in_pdf, file_type)
            elif file_type in ("ms-wordx"):
                return self.docx_to_text(file_path, file_type)
            else:
                err_msg = f"Unsupported file extension: {file_ext} in file {file_path}"
                raise LlmChatException(err_msg)
        else:
            err_msg = f"File checks failed for file: {file_path}. Error: {err_msg_file_checks}"
            self.logger.warning(err_msg)
            raise LlmChatException(err_msg)


    # def format_for_llm(self, file_path: str, include_metadata: bool = True) -> tuple[str, Dict[str, str]]:
    #     """
    #     Format file content with metadata for LLM consumption.
    #
    #     Args:
    #         file_path: Path to the file
    #         include_metadata: Whether to include metadata in output
    #
    #     Returns:
    #         Formatted string ready for LLM
    #     """
    #     try:
    #         content, metadata = self.convert_file(file_path)
    #
    #         file_type = metadata.get('type', 'unknown')
    #         file_name = metadata.get('name', 'unknown')
    #         mime_type = metadata.get('mime_type', 'unknown')
    #         if file_type == 'text':
    #             # For text files, we can include the content directly
    #             return content, metadata
    #         elif file_type in ('image', 'binary'):
    #             result = f"data:{mime_type};base64,{content}"
    #             return result, metadata
    #         else:
    #             raise LlmChatException(f"Unsupported file type: {file_type} for file {file_name}")
    #     except Exception as e:
    #         err_msg = f"Error formatting file {file_path} for LLM: {e}"
    #         raise LlmChatException(err_msg) from e


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