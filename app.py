import streamlit as st
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.database import DataBase
from src.pipelines.training_pipeline import Pipeline
from src.pipelines.prediction_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging
import sys
import time

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .success-message {
            padding: 1rem;
            border-radius: 5px;
            background-color: #D4EDDA;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 5px;
            background-color: #F8D7DA;
            color: #721C24;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .centered-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #1E88E5, #1565C0);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize pipelines
pipeline = Pipeline()
predict_pipeline = PredictPipeline()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_pipeline_id' not in st.session_state:
    st.session_state.current_pipeline_id = None


def process_document(uploaded_file, pipeline_id, action):
    """Process the uploaded PDF document and perform specified action."""
    try:
        result = 0
        if action == "create":
            result = pipeline.create_pipeline(pipeline_id=int(pipeline_id), docs_file=uploaded_file)
        elif action == "remove":
            result = pipeline.delete_pipeline(int(pipeline_id))
        return result
    except Exception as e:
        raise CustomException(e, sys)


def chat_interface():
    """Handle chat interface and query processing."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}:**\n{source}")

    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get pipeline ID from session state
        if not st.session_state.current_pipeline_id:
            with st.chat_message("assistant"):
                st.write("Please select a pipeline ID first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please select a pipeline ID first."})
            return

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = predict_pipeline.query_pipeline(int(st.session_state.current_pipeline_id), prompt)
                if response == -1:
                    message = "Pipeline not found. Please check the pipeline ID."
                    st.write(message)
                    st.session_state.messages.append({"role": "assistant", "content": message})
                else:
                    st.write(response["answer"])
                    if response.get("sources"):
                        with st.expander("Sources"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {idx}:**\n{source}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })


def main():
    # Header
    st.markdown('<div class="centered-header"><h1>🤖 RAG Assistant</h1></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Pipeline Management")

        # Create Pipeline Section
        with st.expander("Create New Pipeline", expanded=True):
            pipeline_id = st.text_input("Pipeline ID:", key="create_pipeline_id", placeholder="Enter numeric ID")
            uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="create_file")

            if st.button("Create Pipeline", key="create_btn"):
                if uploaded_file and pipeline_id:
                    try:
                        with st.spinner("Creating pipeline..."):
                            result = process_document(uploaded_file, pipeline_id, "create")
                            if result == 1:
                                st.success("Pipeline created successfully!")
                            elif result == -1:
                                st.error("Pipeline ID already exists.")
                            else:
                                st.error("Failed to create pipeline.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please provide both Pipeline ID and PDF document.")

        # Delete Pipeline Section
        with st.expander("Delete Pipeline", expanded=False):
            delete_pipeline_id = st.text_input("Pipeline ID to Delete:", key="delete_pipeline_id")
            if st.button("Delete Pipeline", key="delete_btn"):
                if delete_pipeline_id:
                    try:
                        with st.spinner("Deleting pipeline..."):
                            result = process_document(None, delete_pipeline_id, "remove")
                            if result == 1:
                                st.success("Pipeline deleted successfully!")
                                if st.session_state.current_pipeline_id == delete_pipeline_id:
                                    st.session_state.current_pipeline_id = None
                            else:
                                st.error("Pipeline not found.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please provide a Pipeline ID.")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Active Pipeline")

        # Use callback for pipeline ID change
        def on_pipeline_change():
            st.session_state.current_pipeline_id = st.session_state.active_pipeline_id
            st.session_state.messages = []  # Clear chat history when pipeline changes

        active_pipeline_id = st.text_input(
            "Enter Pipeline ID for Chat:",
            key="active_pipeline_id",
            on_change=on_pipeline_change
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # System Information
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("System Information")
        st.info("Using Llama Model for RAG")
        st.progress(100, "System Ready")
        if st.session_state.current_pipeline_id:
            st.success(f"Active Pipeline: {st.session_state.current_pipeline_id}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        # Chat Interface
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        chat_interface()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()