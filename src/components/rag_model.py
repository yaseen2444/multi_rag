import os
import sys
import torch
from src.logger import logging
from src.exception import CustomException
from langchain.llms import HuggingFacePipeline
from dataclasses import dataclass
from transformers import AutoTokenizer,pipeline,AutoModelForCausalLM

@dataclass
class ModelConfig:
    model_name="meta-llama/Llama-3.2-1B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.75

class RagModel:
    def __init__(self):
        self.model_config=ModelConfig()

    def load_model(self, model_name:str = None):
        try:
            model_name = model_name or self.model_config.model_name

            logging.info(f"Loading model and tokenizer from Hugging Face hub: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {device}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            logging.info("Model and tokenizer loaded successfully.")

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


