import streamlit as st
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.database import DataBase
from src.exception import CustomException
from src.logger import logging
import sys

def process_document(uploaded_file, pipeline_id, action):
    """
    Process the uploaded PDF document and perform specified action.

    Args:
        uploaded_file: The uploaded PDF file object.
        pipeline_id: Pipeline ID provided by the user.
        action: Action to perform ("create" or "add").
    """
    try:
        data_ingestion = DataIngestion()
        data_transform = DataTransformation()

        # Save and process the uploaded file
        storage_path = data_ingestion.initiate_ingestion(
            file=uploaded_file,
            pipeline_id=int(pipeline_id)
        )
        chunks, embeddings = data_transform.process_pdf(storage_path)

        # Perform the specified action
        db = DataBase()
        if action == "create":
            db.create_database(pipeline_id, chunks, embeddings)
        elif action == "add":
            db.add_data(chunks,pipeline_id, embeddings)

        return storage_path
    except CustomException as e:
        raise e
    except Exception as e:
        raise CustomException(e, sys)


def main():
    st.title("PDF RAG Application with LangChain and Llama")

    # Sidebar: Upload and Process
    st.sidebar.header("Upload and Process")

    pipeline_id = st.sidebar.text_input("Enter Pipeline ID:", key="pipeline_input")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"], key="file_upload")

    # Create RAG Chatbot Button
    if st.button("Create RAG Chatbot"):
        if uploaded_file and pipeline_id:
            with st.spinner("Processing document for RAG Chatbot..."):
                try:
                    storage_path = process_document(uploaded_file, pipeline_id, action="create")
                    st.success(f"Document processed and stored successfully at: {storage_path}")
                except CustomException as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning("Please upload a file and enter a Pipeline ID.")

    # Add Data Button
    pipeline_id2 = st.sidebar.text_input("Enter Pipeline ID for Add Data:", key="pipeline_input2")
    uploaded_file2 = st.sidebar.file_uploader("Upload a PDF Document for Add Data", type=["pdf"], key="file_upload2")

    if st.button("Add Data"):
        if uploaded_file2 and pipeline_id2:
            with st.spinner("Adding data to the database..."):
                try:
                    storage_path = process_document(uploaded_file2, pipeline_id2, action="add")
                    st.success(f"Data added successfully from document: {uploaded_file2.name}")
                except CustomException as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning("Please upload a file and enter a Pipeline ID.")


if __name__ == "__main__":
    main()
