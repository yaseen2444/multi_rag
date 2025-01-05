from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
pipelines: Dict[str, Any] = {}
PERSIST_DIR = "chroma_db"
KEYS_FILE = "keys.txt"

# Ensure directories exist
os.makedirs(PERSIST_DIR, exist_ok=True)
if not os.path.exists(KEYS_FILE):
    open(KEYS_FILE, 'a').close()


def load_pipeline(pipeline_id: str) -> dict:
    """Load a specific pipeline by ID."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR + pipeline_id,
            embedding_function=embeddings
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm_model(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            verbose=True
        )

        pipeline_data = {
            "chain": chain,
            "vectorstore": vectorstore
        }
        pipelines[pipeline_id] = pipeline_data
        logger.info(f"Successfully loaded pipeline: {pipeline_id}")
        return pipeline_data

    except Exception as e:
        logger.error(f"Error loading pipeline {pipeline_id}: {str(e)}")
        raise


def pipeline_exists(pipeline_id: str) -> bool:
    """Check if a pipeline exists in the keys file."""
    with open(KEYS_FILE, 'r') as file:
        return any(line.strip() == pipeline_id for line in file)


def llm_model():
    """Initialize and return the LLM model."""
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        num_return_sequences=1
    )
    return HuggingFacePipeline(pipeline=pipe)


def process_pdf(file_path):
    """Process PDF file and return document chunks."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise


class QueryRequest(BaseModel):
    question: str


@app.post("/create_pipeline/{pipeline_id}")
async def create_pipeline(pipeline_id: str, file: UploadFile = File(...)):
    if not pipeline_id.strip():
        raise HTTPException(status_code=400, detail="Pipeline ID cannot be empty")

    if pipeline_exists(pipeline_id):
        raise HTTPException(
            status_code=400,
            detail="Pipeline ID already exists. Please select another ID."
        )

    file_path = f"temp_{file.filename}"
    try:
        # Save and process the uploaded file
        with open(file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        docs = process_pdf(file_path)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create and persist the vectorstore
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR + pipeline_id
        )

        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm_model(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            verbose=True
        )

        # Save pipeline to memory and persist ID
        pipelines[pipeline_id] = {
            "chain": chain,
            "vectorstore": vectorstore
        }
        with open(KEYS_FILE, "a") as file_key:
            file_key.write(f"{pipeline_id}\n")

        return {"message": "Pipeline created successfully"}

    except Exception as e:
        logger.error(f"Pipeline creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/append_data/{pipeline_id}")
async def append_data(pipeline_id: str, file: UploadFile = File(...)):
    """Add more data to an existing pipeline."""
    try:
        if not pipeline_exists(pipeline_id):
            raise HTTPException(
                status_code=404,
                detail="Pipeline not found"
            )

        # Get or load the pipeline
        pipeline_data = pipelines.get(pipeline_id)
        if not pipeline_data:
            pipeline_data = load_pipeline(pipeline_id)

        # Process new PDF
        file_path = f"temp_{file.filename}"
        try:
            with open(file_path, "wb") as temp_file:
                content = await file.read()
                temp_file.write(content)

            new_docs = process_pdf(file_path)

            # Add new documents to existing vectorstore
            vectorstore = pipeline_data["vectorstore"]
            vectorstore.add_documents(new_docs)

            # Update the chain with new retriever
            chain = RetrievalQA.from_chain_type(
                llm=llm_model(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                return_source_documents=True,
                verbose=True
            )

            # Update pipeline in memory
            pipelines[pipeline_id] = {
                "chain": chain,
                "vectorstore": vectorstore
            }

            return {"message": "Data appended successfully"}

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        logger.error(f"Append data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/{pipeline_id}")
async def query_pipeline(pipeline_id: str, query: QueryRequest):
    try:
        if not pipeline_exists(pipeline_id):
            raise HTTPException(
                status_code=404,
                detail="Pipeline not found"
            )

        # Get or load the pipeline
        pipeline_data = pipelines.get(pipeline_id)
        if not pipeline_data:
            pipeline_data = load_pipeline(pipeline_id)

        result = pipeline_data["chain"]({"query": query.question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result.get("source_documents", [])]
        }

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/pipeline/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    try:
        # Remove from memory
        if pipeline_id in pipelines:
            del pipelines[pipeline_id]

        # Remove from keys file
        with open(KEYS_FILE, 'r') as file:
            lines = file.readlines()
        with open(KEYS_FILE, 'w') as file:
            file.writelines(line for line in lines if line.strip() != pipeline_id)

        # Remove vectorstore directory
        import shutil
        pipeline_dir = PERSIST_DIR + pipeline_id
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)

        return {"message": "Pipeline deleted successfully"}

    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))