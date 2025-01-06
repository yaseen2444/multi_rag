import os
import sys

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings

from src.utils import validate_file_path


@dataclass
class DataTransformationConfig:
    model_name:str ="sentence-transformers/all-MiniLM-L6-v2"
    chunk_size :int = 1000
    chunk_overlap:int = 200

class DataTransformation:
    def __init__(self):
        self.transform_config=DataTransformationConfig()

    def load_data(self,path:str):
        """
                Load a PDF document and split it into chunks.
                Args:
                    path: Path to the PDF file
                Returns:
                    List of document chunks
                Raises:
                    CustomException: If document loading or splitting fails
                """
        try:
            loader=PyPDFLoader(path)
            docs=loader.load()
            logging.info("data has been loaded")
            splitter=RecursiveCharacterTextSplitter(
                chunk_size=self.transform_config.chunk_size,
                chunk_overlap=self.transform_config.chunk_overlap
            )
            chunks = splitter.split_documents(
                documents=docs
            )
            logging.info("data has been splitted to chunks")
            return chunks
        except Exception as e:
            raise CustomException(e,sys)
    def transform_data(self):
        """
                Initialize and return the embedding model.
                Returns:
                    Configured HuggingFace embeddings model
                Raises:
                    CustomException: If embedding model initialization fails
                """
        try:
            logging.info("loading embedding model")
            embeddings=HuggingFaceEmbeddings(
                model_name=self.transform_config.model_name,
               encode_kwargs = {"normalize_embeddings": True}
            )
            logging.info("embedding model loaded")
            return embeddings

        except Exception as e:

            raise CustomException(e,sys)
    def process_pdf(self,path:str):
        """
        Complete document processing pipeline.
        Args:
            file_path: Path to the PDF file
        Returns:
            Tuple of (document chunks, embeddings model)
        """
        try:
            logging.info("validating the file path")
            if validate_file_path(path):
                logging.info("file path validated")
                chunks=self.load_data(path)
                text_embedding=self.transform_data()

                return chunks,text_embedding
            else:
                logging.info("file path not found")
                raise FileNotFoundError(f"Document not found at: {path}")
        except Exception as e:
            raise CustomException(e,sys)
