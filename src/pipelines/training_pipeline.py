import os
import sys
import torch
from typing import Dict, Any
from src.components.rag_model import RagModel
from src.logger import logging
from src.exception import CustomException
from src.utils import pipeline_exists
from langchain.chains import RetrievalQA
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.database import DataBase


@dataclass
class TrainConfig:
    key_path: str = os.path.join(
        os.getcwd(),"RAG_BUILDER", "artifacts", "keys.txt"
    )
    search_kwargs = {"k": 2}
    verbose: bool = True
    return_source_documents: bool = False

class Pipeline:
    def __init__(self):
        self.train_config = TrainConfig()
        self.pipeline_dict = {}

    def create_pipeline(self, pipeline_id: int, docs_file) -> int:
        """
        Create a new pipeline for document processing and QA

        Args:
            pipeline_id: Unique identifier for the pipeline
            docs_file: Document file to process

        Returns:
            1 if successful, -1 if pipeline already exists, -2 for other errors
        """
        try:
            # Validate inputs
            if not pipeline_id or not docs_file:
                logging.error("Invalid pipeline_id or docs_file")
                return -2

            try:
                if pipeline_exists(str(pipeline_id)):
                    logging.warning(f"Pipeline {pipeline_id} already exists")
                    return -1
            except Exception as e:
                logging.error(f"Error checking pipeline existence: {str(e)}")
                return -2

            # Initialize components
            data_ingestion = DataIngestion()
            data_transform = DataTransformation()
            model = RagModel()

            # Process document
            storage_path = data_ingestion.initiate_ingestion(
                file=docs_file,
                pipeline_id=pipeline_id
            )

            if not storage_path or not os.path.exists(storage_path):
                logging.error("Document storage failed")
                return -2

            chunks, embeddings = data_transform.process_pdf(storage_path)

            # Create database and chain
            db = DataBase()
            vector_store = db.create_database(pipeline_id, chunks, embeddings)

            chain = RetrievalQA.from_chain_type(
                llm=model.load_model(),
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs=self.train_config.search_kwargs
                ),
                return_source_documents=self.train_config.return_source_documents,
                verbose=self.train_config.verbose
            )

            # Save pipeline
            self.pipeline_dict[pipeline_id] = {
                "chain": chain,
                "vectorstore": vector_store
            }

            # Persist pipeline ID
            with open(self.train_config.key_path, "a") as file_key:
                file_key.write(f"\n{pipeline_id}\n")

            logging.info(f"Successfully created pipeline {pipeline_id}")
            return 1

        except Exception as e:
            logging.error(f"Error creating pipeline: {str(e)}")
            raise CustomException(e, sys)

    def delete_pipeline(self, pipeline_id: int) -> int:
        """
        Delete an existing pipeline

        Args:
            pipeline_id: ID of pipeline to delete

        Returns:
            1 if successful, -1 if pipeline doesn't exist
        """
        try:
            # Check if pipeline exists
            if not pipeline_exists(str(pipeline_id)):
                logging.warning(f"Pipeline {pipeline_id} does not exist")
                return -1

            # Remove from pipeline dictionary if present
            if pipeline_id in self.pipeline_dict:
                del self.pipeline_dict[pipeline_id]

            # Remove from keys file
            with open(self.train_config.key_path, 'r') as file:
                lines = file.readlines()
            with open(self.train_config.key_path, 'w') as file:
                file.writelines(line for line in lines if line.strip() != str(pipeline_id))

            # Remove vectorstore directory
            db = DataBase()
            db.remove_database(pipeline_id)

            logging.info(f"Successfully deleted pipeline {pipeline_id}")
            return 1

        except Exception as e:
            logging.error(f"Error deleting pipeline: {str(e)}")
            raise CustomException(e, sys)