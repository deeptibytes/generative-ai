# Enhanced Q&A Chatbot with OpenAI

## Project Overview
This is an interactive Q&A chatbot application built with Streamlit and LangChain that leverages OpenAI's powerful language models. The chatbot provides intelligent real-time responses to user queries with customizable parameters for controlling response behavior and model selection.

## Features
- **Multiple OpenAI Models**: Support for GPT-4o, GPT-4-Turbo, and GPT-4
- **Interactive UI**: Built with Streamlit for an intuitive and responsive user interface
- **Adjustable Parameters**:
  - Temperature control to adjust response creativity (0.0 - 1.0)
  - Max Tokens slider to control response length (50 - 300 tokens)
- **LangChain Integration**: Uses LangChain for prompt management and LLM orchestration
- **LangSmith Tracking**: Built-in LangSmith integration for request tracing and debugging
- **Secure API Key Input**: Password-protected API key input field

## Prerequisites
- Python 3.8+
- OpenAI API account with valid API key
- Required Python packages (see Installation section)

## Installation & Setup

### Step 1: Get OpenAI API Key
1. Visit https://platform.openai.com/settings/organization/api-keys
2. Sign in with your OpenAI account (create one if needed)
3. Click "Create new secret key"
4. Copy and save your API key securely

### Step 2: Install Python Dependencies
Install the required Python packages:
```bash
pip install streamlit langchain-openai python-dotenv
```

### Step 3: Set Up Environment Variables (Optional)
Create a `.env` file in the project directory to store your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

## Running the Application
To start the chatbot application, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage
1. **Enter API Key**: Paste your OpenAI API key in the "Enter your Open AI API Key" field on the sidebar
2. **Select Model**: Choose from the available OpenAI models:
   - GPT-4o (recommended for best performance)
   - GPT-4-Turbo
   - GPT-4
3. **Adjust Parameters**:
   - Move the Temperature slider to control response creativity (0.0 = deterministic, 1.0 = creative)
   - Adjust Max Tokens to control response length
4. **Ask Questions**: Type your question in the input field and press Enter
5. **View Response**: The chatbot will generate and display the response

## How It Works
The chatbot uses a chain of LangChain components:
- **Prompt Template**: Formats user questions with a helpful system prompt
- **ChatOpenAI LLM**: Executes inference using the selected OpenAI model
- **Output Parser**: Extracts and formats the model's response

## Technologies Used
- **Streamlit**: Web application framework
- **LangChain**: LLM framework and orchestration
- **OpenAI API**: Advanced language models (GPT-4, etc.)
- **Python**: Programming language
- **LangSmith**: Tracing and debugging (optional)

## API Costs
Please note that using OpenAI's API incurs charges based on token usage. Monitor your usage on the OpenAI dashboard to manage costs.

## Troubleshooting
- **"Invalid API Key"**: Ensure your API key is correctly copied and has appropriate permissions
- **Rate Limiting**: If you hit rate limits, wait before making additional requests
- **No Response**: Check your internet connection and verify the model is available in your region
