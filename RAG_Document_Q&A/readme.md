# RAG Document Q&A with Groq and Llama 3

## Project Overview
This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that enables intelligent question-answering over PDF documents. The system combines OpenAI embeddings with Groq's fast Llama 3.1 language model to provide accurate, context-aware answers based on the content of your research papers and documents.

## Features
- **PDF Document Processing**: Automatically loads and processes multiple PDF files from a local directory
- **Vector Embeddings**: Uses OpenAI embeddings to create semantic representations of document chunks
- **FAISS Vector Store**: Implements Facebook's FAISS for efficient similarity search
- **RAG Pipeline**: Combines retrieval and generation for accurate, context-grounded responses
- **Groq Llama 3.1**: Leverages Groq's optimized Llama 3.1-8B model for fast inference
- **Interactive UI**: Clean Streamlit interface with document similarity display
- **Session State Management**: Efficient caching of embeddings and vector store
- **Response Time Tracking**: Monitors and logs query processing time

## How It Works
The application implements a complete RAG (Retrieval-Augmented Generation) workflow:

1. **Document Ingestion**: Loads PDF files from the `research_papers` directory
2. **Text Chunking**: Splits documents into overlapping chunks (1000 characters, 200 character overlap)
3. **Embedding Creation**: Converts text chunks to semantic embeddings using OpenAI
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Query Processing**: When a user asks a question:
   - Retrieves relevant document chunks using similarity search
   - Sends the context and query to Groq's Llama 3.1 model
   - Returns context-aware answer with supporting document excerpts

## Prerequisites
- Python 3.8+
- Groq API key (https://console.groq.com)
- OpenAI API key (https://platform.openai.com/api-keys)
- Research papers or documents in PDF format
- Required Python packages (see Installation section)

## Installation & Setup

### Step 1: Install Python Dependencies
Install the required packages:
```bash
pip install streamlit langchain-groq langchain-openai langchain-community faiss-cpu PyPDF2 python-dotenv
```

For GPU acceleration with FAISS (optional):
```bash
pip install faiss-gpu
```

### Step 2: Set Up API Keys
Create a `.env` file in the project directory with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API keys from:
- **Groq API**: https://console.groq.com
- **OpenAI API**: https://platform.openai.com/api-keys

### Step 3: Prepare Your Documents
1. Create a folder named `research_papers` in the project directory:
```bash
mkdir research_papers
```

2. Place your PDF documents in this folder:
```bash
cp /path/to/your/documents/*.pdf research_papers/
```

## Running the Application

### Step 1: Navigate to Project Directory
```bash
cd /path/to/RAG_Document_Q&A
```

### Step 2: Run the Streamlit Application
```bash
streamlit run main.py
```

### Step 3: Access the Web Interface
The application will automatically open in your default browser at `http://localhost:8501`

### Step 4: Using the Application
1. **Create Vector Embeddings**:
   - Click the "Document Embedding" button to load your PDFs and create the vector store
   - Wait for the message "Vector Database is ready"
   - This step processes the first 50 documents for performance

2. **Ask Questions**:
   - Enter your question in the text input field: "Enter your query from the research paper"
   - The system will retrieve relevant document chunks and generate an answer
   - Processing time will be displayed

3. **View Source Documents**:
   - Click on "Document similarity Search" expander to view the source document chunks
   - Each relevant chunk is displayed with clear separators

## Key Technologies

### LangChain Components
- **ChatPromptTemplate**: Defines the prompt structure for document-based Q&A
- **RecursiveCharacterTextSplitter**: Splits documents into manageable chunks
- **create_stuff_documents_chain**: Combines retrieved documents with the prompt
- **create_retrieval_chain**: Orchestrates the retrieval and generation pipeline
- **PyPDFDirectoryLoader**: Loads PDF files from a directory

### Embeddings & Vector Store
- **OpenAIEmbeddings**: Generates semantic embeddings for document chunks
- **FAISS**: Efficient similarity search in vector space

### Language Model
- **ChatGroq (Llama 3.1-8B-Instant)**: Fast, efficient language model for generating responses


**Last Updated**: January 25, 2026
