# Enhanced Q&A Chatbot with Ollama

## Project Overview
This is an interactive Q&A chatbot application built with Streamlit and LangChain that uses the Ollama framework to run open-source language models locally. The chatbot provides real-time responses to user queries with adjustable parameters for fine-tuning the model's behavior.


## Prerequisites
- Python 3.8+
- Ollama installed on your system
- Required Python packages (see Installation section)

## Installation & Setup

### Step 1: Download and Install Ollama
Visit https://ollama.com/download/mac and download the Ollama application for macOS. Install it by following the installation wizard.

### Step 2: Pull the Mistral Model
After installing Ollama, run the following command to download the Mistral model:
```bash
ollama pull mistral
```
Wait for the download to complete.

### Step 3: Install Python Dependencies
Install the required Python packages:
```bash
pip install streamlit langchain-openai langchain-community python-dotenv
```

### Step 4: Set Up Environment Variables (Optional)
Create a `.env` file in the project directory and add your LangChain API key if you want to enable tracing:
```
LANGCHAIN_API_KEY=your_api_key_here
```

## Running the Application
To start the chatbot application, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`


## How It Works
The chatbot uses a chain of LangChain components:
- **Prompt Template**: Formats user questions with a helpful system prompt
- **Ollama LLM**: Executes the inference using the selected model
- **Output Parser**: Extracts and formats the model's response

## Technologies Used
- **Streamlit**: Web application framework
- **LangChain**: LLM framework and orchestration
- **Ollama**: Local inference engine for open-source models
- **Python**: Programming language

## Notes
- Ensure Ollama is running before starting the Streamlit app
- The first request may take longer as the model initializes
- Response quality and speed depend on your system's hardware capabilities
