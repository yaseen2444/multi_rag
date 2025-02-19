import os
import sys
from src.logger import logging
from src.exception import CustomException
KEYS_FILE=os.path.join("RAG_BUILDER",'artifacts','keys.txt')

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

def pipeline_exists(pipeline_id: str) -> bool:
    """Check if a pipeline exists in the keys file."""
    with open(KEYS_FILE, 'r') as file:
        return any(line.strip() == pipeline_id for line in file)