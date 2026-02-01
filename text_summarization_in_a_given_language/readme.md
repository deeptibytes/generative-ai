# Text Summarization and Translation in Specified Language

## Project Overview
This Jupyter notebook demonstrates how to use LangChain and Groq's Llama 3.1 model to summarize a speech and translate the summary into a specified language using Devanagari script. The example uses a Hindi speech about Indian government schemes and translates the summary to Hindi.

## Features
- **Speech Summarization**: Uses Groq's Llama 3.1 model to generate concise summaries of long text
- **Language Translation**: Translates the summary to a specified language (Hindi in the example)
- **Devanagari Script Output**: Ensures the translated text uses native Devanagari script
- **Token Counting**: Displays the number of tokens in the input speech
- **LangChain Integration**: Uses LLMChain for prompt management and execution

## Step-by-Step Implementation Details

### 1. **Environment Setup**
```python
import os
from dotenv import load_dotenv
load_dotenv()
```
- Imports necessary modules for environment variable handling
- Loads environment variables from `.env` file

### 2. **LLM Configuration**
```python
from langchain_groq import ChatGroq
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant")
```
- Imports ChatGroq from LangChain
- Retrieves Groq API key from environment variables
- Initializes the Llama 3.1 model for text generation

### 3. **Input Speech**
The notebook contains a long Hindi speech about:
- Viksit Bharat Sankalp Yatra (Developed India Resolution Journey)
- Government schemes like Pradhan Mantri Awas Yojana
- Experiences with Ayushman card and other welfare programs
- Impact on government officers and beneficiaries

### 4. **Token Analysis**
```python
llm.get_num_tokens(speech)
```
- Calculates the number of tokens in the speech text
- Helps understand input size for the model

### 5. **Prompt Template Creation**
```python
from langchain.chains import LLMChain
from langchain import PromptTemplate

generictemplate = """
Write a summary of the following speech:
Speech:{speech}
Translate the precise summary to {language}
Use native Devanagari script. Output ONLY the translated text
"""

prompt = PromptTemplate(
    input_variables=['speech','language'],
    template=generictemplate
)
```
- Defines a prompt template for summarization and translation
- Uses placeholders for speech content and target language
- Instructs the model to output only the translated text in Devanagari script

### 6. **Chain Execution**
```python
llm_chain = LLMChain(llm=llm, prompt=prompt)
summary = llm_chain.invoke({'speech': speech, 'language': 'Hindi'})
print(summary["text"])
```
- Creates an LLMChain with the configured LLM and prompt
- Invokes the chain with the speech and language parameters
- Prints the translated summary in Hindi

**Last Updated**: January 31, 2026
