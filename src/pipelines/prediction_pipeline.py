import os
import sys
from typing import Union, Dict, Any

from langchain.chains import RetrievalQA
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.database import DataBase
from src.components.rag_model import RagModel
from src.utils import pipeline_exists


class PredictPipeline:
    def __init__(self):
        self.data_transform = DataTransformation()
        self.data_base = DataBase()
        self.model = RagModel()
        self.search_kwargs = {"k": 2}
        self.pipelines: Dict[int, Any] = {}

    def _load_pipeline(self, pipeline_id: int) -> Dict[str, Any]:
        """
        Load or get an existing pipeline

        Args:
            pipeline_id: Unique identifier for the pipeline

        Returns:
            Dict containing the chain and vectorstore

        Raises:
            CustomException: If loading fails
        """
        try:
            # Check if pipeline is already loaded in memory
            if pipeline_id in self.pipelines:
                logging.info(f"Using existing pipeline {pipeline_id} from memory")
                return self.pipelines[pipeline_id]

            logging.info(f"Loading pipeline {pipeline_id} from disk")

            # Get embeddings from data transformation
            embeddings = self.data_transform.transform_data()

            # Load the vector store
            vectorstore = self.data_base.load_database(pipeline_id, embeddings)

            # Create the chain
            chain = RetrievalQA.from_chain_type(
                llm=self.model.load_model(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs=self.search_kwargs),
                return_source_documents=True,
                verbose=True
            )

            # Store in memory
            pipeline_data = {
                "chain": chain,
                "vectorstore": vectorstore
            }
            self.pipelines[pipeline_id] = pipeline_data

            logging.info(f"Successfully loaded pipeline {pipeline_id}")
            return pipeline_data

        except Exception as e:
            logging.error(f"Error loading pipeline {pipeline_id}: {str(e)}")
            raise CustomException(e, sys)

    def query_pipeline(self, pipeline_id: int, query: str) -> Union[Dict[str, Any], int]:
        """
        Query a specific pipeline with a question

        Args:
            pipeline_id: Unique identifier for the pipeline
            query: Question to ask

        Returns:
            Dict containing answer and sources, or -1 if pipeline doesn't exist

        Raises:
            CustomException: If query processing fails
        """
        try:
            # Validate pipeline exists
            if not pipeline_exists(str(pipeline_id)):
                logging.warning(f"Pipeline {pipeline_id} does not exist")
                return -1

            # Load or get pipeline
            pipeline_data = self._load_pipeline(pipeline_id)

            # Process query
            logging.info(f"Processing query for pipeline {pipeline_id}")
            result = pipeline_data["chain"]({"query": query})

            # Format response
            response = {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result.get("source_documents", [])]
            }

            logging.info("Query processed successfully")
            return response

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise CustomException(e, sys)