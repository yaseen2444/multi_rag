import os
import sys
from src.logger import logging
from src.exception import CustomException


def validate_file_path(file_path: str) -> bool:
    """
    Validate if the file path exists and is a PDF.

    Args:
        file_path: Path to validate

    Returns:
        bool indicating if path is valid
    """
    return (
            os.path.exists(file_path) and
            os.path.isfile(file_path) and
            file_path.lower().endswith('.pdf')
    )
