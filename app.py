import streamlit as st
import requests
import tempfile
import os


def main():
    st.title("PDF RAG Application with LangChain and Llama")

    st.sidebar.header("Upload and Process")
    pipeline_id = st.sidebar.text_input("Enter Pipeline ID:", key="pipeline_input")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

    if "pipeline_id" not in st.session_state:
        st.session_state.pipeline_id = None

    if uploaded_file and pipeline_id:
        st.write("### Uploaded Document")
        st.write(f"**File Name:** {uploaded_file.name}")

        if st.button("Create RAG Chatbot"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"http://localhost:8080/create_pipeline/{pipeline_id}",
                files=files
            )

            if response.status_code == 200:
                st.session_state.pipeline_id = pipeline_id
                st.success("RAG chatbot created successfully!")
            else:
                st.error(f"Failed to create RAG chatbot: {response.text}")

    if st.session_state.pipeline_id:
        st.write("### Chat with your document")
        query = st.text_input("Enter your question:")

        if query:
            response = requests.post(
                f"http://localhost:8080/query/{st.session_state.pipeline_id}",
                json={"question": query}
            )

            if response.status_code == 200:
                result = response.json()
                st.write("**Answer:**", result["answer"])

                with st.expander("View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.write(f"Source {i}:", source)


if __name__ == "__main__":
    main()