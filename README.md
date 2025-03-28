# RAG Matrix: Multi-Pipeline Document Chat Application

## 🤖 Project Overview

RAG Matrix is a sophisticated web application that allows users to create, manage, and interact with multiple Retrieval-Augmented Generation (RAG) pipelines. Users can upload PDF documents, create unique pipeline IDs, and chat with their custom AI assistants trained on specific document sets.

### Key Features
- 📄 Multiple Pipeline Management
- 🔍 Document-Based AI Chat
- 🔐 Unique Pipeline IDs
- 🖥️ Intuitive Streamlit Interface
- 🐍 Modular Python Architecture

## 🚀 Project Structure

```
rag_builder/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── database.py
│   │   └── ...
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── ...
│
├── app.py
├── setup.py
└── requirements.txt
```

## 🛠️ Technologies Used
- Streamlit
- LangChain
- ChromaDB
- Transformers
- PyTorch
- FastAPI
- Sentence Transformers

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
```bash
git clone https://github.com/yaseen2444/rag-matrix.git
cd rag-matrix
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## 🖥️ Running the Application
```bash
streamlit run app.py
```

## 💡 How to Use

### Creating a Pipeline
1. Enter a unique Pipeline ID
2. Upload a PDF document
3. Click "Create Pipeline"

### Chatting with Your RAG
1. Select an existing Pipeline ID
2. Type your question in the chat input
3. Receive AI-generated responses based on your document

## 🔍 Debug Mode
- Toggle the "Debug Mode" checkbox in the sidebar
- Get additional system information and insights

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact
Mail - yaseenmohammadap37@gmail.com

