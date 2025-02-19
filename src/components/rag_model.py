from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login, snapshot_download
from langchain.llms import HuggingFacePipeline
from dataclasses import dataclass
import torch
import sys
import os
from src.logger import logging
from src.exception import CustomException

hugging_face_token = "enter you token"
login(token=hugging_face_token)

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.75
    cache_dir: str = "model_cache"  # Directory for storing downloaded models

class RagModel:
    def __init__(self):
        self.model_config = ModelConfig()
        os.makedirs(self.model_config.cache_dir, exist_ok=True)
        
    def download_model(self, model_name: str) -> str:
        """Download the model files to local cache directory."""
        try:
            logging.info(f"Downloading model to local cache: {model_name}")
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.model_config.cache_dir,
                use_auth_token=True
            )
            logging.info(f"Model downloaded successfully to: {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"Error downloading model: {str(e)}")
            raise CustomException(e, sys)

    def load_model(self, model_name: str = None):
        try:
            model_name = model_name or self.model_config.model_name
            
            # Download model to local cache if not already present
            local_model_path = self.download_model(model_name)
            
            logging.info(f"Loading model and tokenizer from local cache: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                use_auth_token=True,
                local_files_only=True  # Only use local files
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {device}")

            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                use_auth_token=True,
                local_files_only=True  # Only use local files
            )

            logging.info("Model and tokenizer loaded successfully from local cache.")

            pipe = pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                max_new_tokens=self.model_config.max_new_tokens,
                temperature=self.model_config.temperature
            )

            logging.info("Pipeline loaded successfully, model creation is complete.")
            return HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            logging.error(f"Error in loading model: {str(e)}")
            raise CustomException(e, sys)

    def clear_cache(self):
        """Clear the local model cache directory."""
        try:
            import shutil
            if os.path.exists(self.model_config.cache_dir):
                shutil.rmtree(self.model_config.cache_dir)
                os.makedirs(self.model_config.cache_dir)
                logging.info("Model cache cleared successfully.")
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")
            raise CustomException(e, sys)