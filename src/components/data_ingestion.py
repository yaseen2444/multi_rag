import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
from typing import List, Optional

@dataclass
class DataIngestionConfig:
    base_path: str = os.path.join("RAG_BUILDER","artifacts", "ingestion")
    max_files: int = 1000  # Increased from default 5
    batch_size: int = 50   # Process files in batches
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        self.processed_files: List[str] = []
        
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
            # Check if we've hit the file limit
            if len(self.processed_files) >= self.config.max_files:
                logging.warning(f"File limit ({self.config.max_files}) reached. Consider increasing max_files in config.")
                return None
            
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
            
            # Add to processed files list
            self.processed_files.append(storage_path)
            
            logging.info(f"File successfully stored at: {storage_path}")
            logging.info(f"Total files processed: {len(self.processed_files)}")
            
            return storage_path
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def batch_process_files(self, files: List, pipeline_id: int) -> List[str]:
        """
        Process multiple files in batches.
        
        Args:
            files: List of StreamlitUploadedFile objects
            pipeline_id: Unique identifier for the pipeline
            
        Returns:
            List[str]: Paths of successfully processed files
        """
        processed_paths = []
        
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]
            
            for file in batch:
                path = self.initiate_ingestion(file, pipeline_id)
                if path:
                    processed_paths.append(path)
            
            logging.info(f"Processed batch {i//self.config.batch_size + 1}")
        
        return processed_paths
    
    def clear_processed_files(self):
        """Reset the processed files counter"""
        self.processed_files = []
        
    def get_processed_count(self) -> int:
        """Get the number of processed files"""
        return len(self.processed_files)