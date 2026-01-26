# Basic Chatbot with Ollama

## Project Overview
This is a simple yet effective Q&A chatbot application built with Streamlit and LangChain that uses the Ollama framework to run the lightweight Gemma 2B open-source language model locally. Perfect for quick responses without relying on cloud APIs.


## Installation & Setup

### Step 1: Download and Install Ollama
Visit https://ollama.com/download/mac and download the Ollama application for macOS. Install it by following the installation wizard.

### Step 2: Pull the Gemma 2B Model
After installing Ollama, run the following command to download the Gemma 2B model:
```bash
ollama pull gemma:2b
```
Wait for the download to complete. The Gemma 2B model is lightweight (approximately 1.4GB) and suitable for most systems.

### Step 3: Install Python Dependencies
Install the required Python packages:
```bash
pip install streamlit langchain-community python-dotenv
```

### Step 4: Set Up Environment Variables (Optional)
Create a `.env` file in the project directory if you want to enable LangSmith tracing:
```
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=your_project_name_here
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
- **Ollama Gemma 2B LLM**: Executes the inference using the local Gemma model
- **Output Parser**: Extracts and formats the model's response

## Technologies Used
- **Streamlit**: Web application framework
- **LangChain**: LLM framework and orchestration
- **Ollama**: Local inference engine for open-source models
- **Gemma 2B**: Lightweight open-source language model by Google
- **Python**: Programming language

## Notes
- Ensure Ollama is running before starting the Streamlit app
- The first request may take a few seconds as the model initializes
- Response quality is optimized for speed due to the lightweight nature of Gemma 2B
- For more complex queries, consider using larger models through the Enhanced Chatbot variant
