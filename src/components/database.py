import os
import sys
import shutil
from dataclasses import dataclass
from typing import Optional

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
from src.utils import validate_file_path


@dataclass
class DataBaseConfig:
    """Configuration for database persistence"""
    PERSIST_DIR: str = os.path.join("artifacts", "chroma_db")


class DataBase:
    """Handles vector database operations while maintaining existing structure"""

    def __init__(self):
        """Initialize database with configuration"""
        self.data_base = DataBaseConfig()
        os.makedirs(self.data_base.PERSIST_DIR, exist_ok=True)
        logging.info(f"Initialized database with persist directory: {self.data_base.PERSIST_DIR}")

    def get_persist_dir(self, pipeline_id: int) -> str:
        """
        Get the persistence directory for a specific pipeline

        Args:
            pipeline_id: Unique identifier for the pipeline

        Returns:
            str: Full path for the pipeline's storage
        """
        logging.info(f"Creating persistent path for pipeline {pipeline_id}")
        persist_path = os.path.join(self.data_base.PERSIST_DIR, str(pipeline_id))
        os.makedirs(persist_path, exist_ok=True)
        return persist_path

    def create_database(self, pipeline_id: int, docs, embeddings: Optional[HuggingFaceEmbeddings]):
        """
        Create a new vector database

        Args:
            pipeline_id: Unique identifier for the pipeline
            docs: Documents to store
            embeddings: Embedding model (optional)

        Returns:
            Chroma: Initialized vector store

        Raises:
            CustomException: If database creation fails
        """
        try:
            final_path = self.get_persist_dir(pipeline_id)

            logging.info("Creating the database")
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=final_path
            )
            vectorstore.persist()
            logging.info(f"Database creation complete for pipeline {pipeline_id}")
            return vectorstore

        except Exception as e:
            logging.error(f"Error in database creation: {str(e)}")
            raise CustomException(e, sys)

    def load_database(self, pipeline_id: int, embeddings: Optional[HuggingFaceEmbeddings]):
        """
        Load an existing vector database

        Args:
            pipeline_id: Unique identifier for the pipeline
            embeddings: Embedding model (optional)

        Returns:
            Chroma: Loaded vector store

        Raises:
            CustomException: If database loading fails
            FileNotFoundError: If database doesn't exist
        """
        try:
            persist_path = self.get_persist_dir(pipeline_id)


            logging.info(f"Loading database for pipeline {pipeline_id}")
            vector_store = Chroma(
                persist_directory=persist_path,
                embedding_function=embeddings
            )
            logging.info("Database loaded successfully")
            return vector_store

        except Exception as e:
            logging.error(f"Error in database loading: {str(e)}")
            raise CustomException(e, sys)

    def add_data(self, additional_docs, pipeline_id: int, embeddings: Optional[HuggingFaceEmbeddings]):
        """
        Add new documents to existing database

        Args:
            additional_docs: New documents to add
            pipeline_id: Unique identifier for the pipeline
            embeddings: Embedding model (optional)

        Returns:
            Chroma: Updated vector store

        Raises:
            CustomException: If data addition fails
        """
        try:
            store = self.load_database(pipeline_id, embeddings)
            logging.info(f"Adding new documents to pipeline {pipeline_id}")

            store.add_documents(additional_docs)
            store.persist()  # Ensure changes are persisted
            logging.info("Data addition successful")

            return store

        except Exception as e:
            logging.error(f"Error in adding data: {str(e)}")
            raise CustomException(e, sys)

    def remove_database(self, pipeline_id: int) -> bool:
        """
        Remove a database and its files

        Args:
            pipeline_id: Unique identifier for the pipeline

        Returns:
            bool: True if removal was successful

        Raises:
            CustomException: If database removal fails
        """
        try:
            persist_path = self.get_persist_dir(pipeline_id)

            if not os.path.exists(persist_path):
                logging.warning(f"No database found at {persist_path}")
                return False

            shutil.rmtree(persist_path)
            logging.info(f"Successfully removed database for pipeline {pipeline_id}")
            return True

        except Exception as e:
            logging.error(f"Error in removing database: {str(e)}")
            raise CustomException(e, sys)



