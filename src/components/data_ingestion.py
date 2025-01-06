import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import sys
@dataclass
class DataIngestionConfig:
    base_path: str = os.path.join("artifacts", "ingestion")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_ingestion(self, file, pipeline_id: int) -> str:
        """
        Process and store uploaded PDF file.

        Args:
            file: StreamlitUploadedFile object
            pipeline_id: Unique identifier for the pipeline

        Returns:
            str: Path where the file was stored

        Raises:
            CustomException: If file processing fails
        """
        try:
            # Create storage path
            storage_path = os.path.join(
                self.config.base_path,
                str(pipeline_id),
                file.name
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

            # Write file contents
            with open(storage_path, 'wb') as f:
                f.write(file.getvalue())

            logging.info(f"File successfully stored at: {storage_path}")
            return storage_path

        except Exception as e:
            raise CustomException(e,sys)
