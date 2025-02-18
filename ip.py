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
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="RAG MATRIX",
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
            background-color: #EEF2F7;
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
            background: linear-gradient(90deg, #EEF2F7, #1565C0);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .chat-container {
            margin-bottom: 60px;
            height: calc(100vh - 300px);
            overflow-y: auto;
        }
        .debug-info {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_pipeline_id' not in st.session_state:
    st.session_state.current_pipeline_id = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

try:
    # Initialize pipelines with error handling
    pipeline = Pipeline()
    predict_pipeline = PredictPipeline()
    initialization_success = True
except Exception as e:
    initialization_success = False
    initialization_error = str(e)
    logging.error(f"Failed to initialize pipelines: {str(e)}")

def process_document(uploaded_file, pipeline_id, action):
    """Process the uploaded PDF document and perform specified action."""
    try:
        logging.info(f"Processing document: action={action}, pipeline_id={pipeline_id}")
        result = 0
        if action == "create":
            result = pipeline.create_pipeline(pipeline_id=int(pipeline_id), docs_file=uploaded_file)
        elif action == "remove":
            result = pipeline.delete_pipeline(int(pipeline_id))
        logging.info(f"Document processing result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in process_document: {str(e)}")
        raise CustomException(e, sys)

def handle_chat(prompt):
    """Handle chat message processing."""
    try:
        logging.info(f"Processing chat prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not st.session_state.current_pipeline_id:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Please select a pipeline ID first."
            })
            return

        with st.spinner("Processing your question..."):
            # Add debug information
            if st.session_state.debug_mode:
                st.info(f"Sending query to pipeline {st.session_state.current_pipeline_id}")
                
            response = predict_pipeline.query_pipeline(int(st.session_state.current_pipeline_id), prompt)
            
            if response == -1:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Pipeline not found. Please check the pipeline ID."
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
                
            logging.info("Chat response processed successfully")
            
    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I encountered an error while processing your request. Error details: {str(e)}"
        })

def main():
    # Header
    st.markdown('<div class="centered-header"><h1>🤖 RAG MATRIX</h1></div>', unsafe_allow_html=True)

    if not initialization_success:
        st.error(f"Failed to initialize the application: {initialization_error}")
        return

    # Debug mode toggle in sidebar
    with st.sidebar:
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.divider()

        st.header("Pipeline Management")

        # Create Pipeline Section
        with st.expander("Create New Pipeline", expanded=True):
            pipeline_id = st.text_input("Pipeline ID:", key="create_pipeline_id", placeholder="Enter numeric ID")
            uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="create_file")

            if st.button("Create Pipeline", key="create_btn"):
                if uploaded_file and pipeline_id:
                    try:
                        with st.spinner("Creating pipeline..."):
                            if st.session_state.debug_mode:
                                st.info(f"Creating pipeline with ID: {pipeline_id}")
                            result = process_document(uploaded_file, pipeline_id, "create")
                            if result == 1:
                                st.success("Pipeline created successfully!")
                            elif result == -1:
                                st.error("Pipeline ID already exists.")
                            else:
                                st.error("Failed to create pipeline.")
                    except Exception as e:
                        st.error(f"Error creating pipeline: {str(e)}")
                else:
                    st.warning("Please provide both Pipeline ID and PDF document.")

        # Delete Pipeline Section
        with st.expander("Delete Pipeline", expanded=False):
            delete_pipeline_id = st.text_input("Pipeline ID to Delete:", key="delete_pipeline_id")
            if st.button("Delete Pipeline", key="delete_btn"):
                if delete_pipeline_id:
                    try:
                        with st.spinner("Deleting pipeline..."):
                            if st.session_state.debug_mode:
                                st.info(f"Deleting pipeline with ID: {delete_pipeline_id}")
                            result = process_document(None, delete_pipeline_id, "remove")
                            if result == 1:
                                st.success("Pipeline deleted successfully!")
                                if st.session_state.current_pipeline_id == delete_pipeline_id:
                                    st.session_state.current_pipeline_id = None
                            else:
                                st.error("Pipeline not found.")
                    except Exception as e:
                        st.error(f"Error deleting pipeline: {str(e)}")
                else:
                    st.warning("Please provide a Pipeline ID.")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        # st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Active Pipeline")

        def on_pipeline_change():
            try:
                st.session_state.current_pipeline_id = st.session_state.active_pipeline_id
                st.session_state.messages = []
                if st.session_state.debug_mode:
                    st.info(f"Switched to pipeline: {st.session_state.current_pipeline_id}")
            except Exception as e:
                st.error(f"Error changing pipeline: {str(e)}")

        active_pipeline_id = st.text_input(
            "Enter Pipeline ID for Chat:",
            key="active_pipeline_id",
            on_change=on_pipeline_change
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # System Information
        # st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("System Information")
        st.info("Using Llama Model for RAG")
        st.progress(100, "System Ready")
        if st.session_state.current_pipeline_id:
            st.success(f"Active Pipeline: {st.session_state.current_pipeline_id}")
        
        if st.session_state.debug_mode:
            st.markdown("### Debug Information")
            st.code(f"""
Current Pipeline ID: {st.session_state.current_pipeline_id}
Messages Count: {len(st.session_state.messages)}
System Status: Active
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat messages display
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {idx}:**\n{source}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input - Placed at the top level
    prompt = st.chat_input("Ask your question...")
    if prompt:
        handle_chat(prompt)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error(f"Application error: {str(e)}\n{traceback.format_exc()}")