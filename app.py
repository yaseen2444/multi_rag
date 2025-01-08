import streamlit as st
import os

from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.database import DataBase
from src.exception import CustomException
from src.logger import logging

def main():
    st.title("PDF RAG Application with LangChain and Llama")

    st.sidebar.header("Upload and Process")
    pipeline_id = st.sidebar.text_input("Enter Pipeline ID:", key="pipeline_input")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file and pipeline_id:
        try:
            st.write("### Uploaded Document")
            st.write(f"**File Name:** {uploaded_file.name}")

            if st.button("Create RAG Chatbot"):
                with st.spinner("Processing document..."):
                    data_ingestion = DataIngestion()
                    data_transform=DataTransformation()
                    storage_path = data_ingestion.initiate_ingestion(
                        file=uploaded_file,
                        pipeline_id=int(pipeline_id)
                    )
                    x,y=data_transform.process_pdf(storage_path)
                    db = DataBase()
                    db.create_database(pipeline_id,x, y)
                    if storage_path:
                        st.success(f"Document processed and stored successfully at: {storage_path}")
                    else:
                        st.error("Failed to process document")
            if st.button("add data"):
                pipeline_id2 = st.sidebar.text_input("Enter Pipeline ID:", key="pipeline_input2")
                uploaded_file2 = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"],key="up2")
                if uploaded_file and pipeline_id:
                    try:
                        st.write("### Uploaded Document")
                        st.write(f"**File Name:** {uploaded_file.name}")
                        data_ingestion = DataIngestion()
                        data_transform = DataTransformation()
                        storage_path = data_ingestion.initiate_ingestion(
                            file=uploaded_file,
                            pipeline_id=int(pipeline_id)
                        )
                        x1, y1= data_transform.process_pdf(storage_path)
                        db=DataBase()
                        db.add_data(pipeline_id,x1,y1)

                    except:
                        pass

        except CustomException as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
