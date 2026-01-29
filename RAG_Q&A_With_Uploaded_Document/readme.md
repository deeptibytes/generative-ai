# Conversational RAG with PDF Uploads and Chat History

## Project Overview
This is an advanced Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that enables multi-turn conversations with PDF documents while maintaining complete chat history. Users can upload multiple PDFs, ask questions about their content, and have follow-up conversations with context awareness across multiple turns. The system uses HuggingFace embeddings and Groq's Llama 3.1 for fast, intelligent responses.

## Key Features
- **PDF Upload & Processing**: Upload and process multiple PDF documents in a single session
- **Conversational Context**: Maintains full chat history across multiple turns
- **History-Aware Retrieval**: Reformulates follow-up questions based on chat history context
- **HuggingFace Embeddings**: Uses lightweight, efficient embeddings for semantic understanding
- **Chroma Vector Store**: Efficient vector database for document similarity search
- **Multi-Session Support**: Handle multiple concurrent chat sessions with unique session IDs
- **Dynamic Prompting**: Intelligent prompt reformulation for context-aware responses
- **Interactive Streamlit UI**: Clean, intuitive web interface with real-time responses

## How It Works - Step-by-Step Implementation

### Step 1: Environment Setup

- Imports all necessary components for document processing, embeddings, and conversational AI

### Step 2: Embeddings Configuration
```python
os.environ['HUGGINGFACE_TOKEN']=os.getenv("HUGGINGFACE_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
- Sets up HuggingFace embeddings using the "all-MiniLM-L6-v2" model
- This lightweight model creates semantic representations of text chunks
- Efficient and suitable for local deployment

### Step 3: User Interface Initialization
```python
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")
```
- Creates the main title and description
- Sets up the Streamlit interface

### Step 4: API Key Input
```python
api_key=st.text_input("Enter your Groq API key:",type="password")
```
- Provides a secure password input field for the Groq API key
- The rest of the app only functions if API key is provided

### Step 5: LLM Initialization
```python
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")
```
- Initializes the Groq ChatGroq model with the provided API key
- Uses Llama 3.1-8B-Instant for fast inference

### Step 6: Session Management
```python
session_id=st.text_input("Session ID",value="default_session")
if 'store' not in st.session_state:
    st.session_state.store={}
```
- Creates a unique session ID for each conversation
- Initializes a session state dictionary to store chat histories
- Enables multiple concurrent conversations

### Step 7: PDF Upload & Processing
```python
uploaded_files=st.file_uploader("Choose A Pdf file",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
```
- Accepts multiple PDF files
- Saves each uploaded file temporarily
- Loads PDF content using PyPDFLoader
- Aggregates all documents into a single list

### Step 8: Document Chunking & Vectorization
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```
- Splits documents into chunks of 5000 characters with 500 character overlap
- Creates vector embeddings using HuggingFace model
- Stores vectors in Chroma for efficient retrieval
- Creates a retriever for semantic search

### Step 9: Context-Aware Question Reformulation
```python
contextualize_q_system_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
```
- Creates a prompt that reformulates follow-up questions
- Uses chat history to understand context ("it", "that", "he/she" references)
- Generates standalone questions for better retrieval
- Creates a history-aware retriever that considers previous conversation

### Step 10: Question-Answering Chain Setup
```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
```
- Defines the system prompt for question-answering
- Creates a prompt template with chat history context
- Sets up constraints (max 3 sentences, concise answers)

### Step 11: RAG Chain Assembly
```python
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
```
- Combines the history-aware retriever with the Q&A chain
- Creates the complete RAG pipeline

### Step 12: Chat History Management
```python
def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]
```
- Function to retrieve or create chat history for a session
- Maintains separate history for each unique session ID

### Step 13: Conversational RAG Chain
```python
conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```
- Wraps the RAG chain with message history management
- Automatically manages chat history for each interaction
- Maintains context across multiple turns

### Step 14: User Interaction Loop
```python
user_input = st.text_input("Your question:")
if user_input:
    session_history=get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id":session_id}},
    )
    st.write("Assistant:", response['answer'])
    st.write("Chat History:", session_history.messages)
```
- Gets user input from text field
- Invokes the conversational RAG chain
- Displays the answer and chat history

## Prerequisites
- Python 3.8+
- Groq API key (https://console.groq.com)
- HuggingFace token (optional, for model access)
- Required Python packages (see Installation section)

## Installation & Setup

### Step 1: Install Python Dependencies
```bash
pip install streamlit langchain langchain-groq langchain-chroma langchain-huggingface langchain-text-splitters PyPDF2 python-dotenv
```

### Step 2: Set Up Environment Variables
Create a `.env` file in the project directory:
```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Get your API keys from:
- **Groq API**: https://console.groq.com
- **HuggingFace Token**: https://huggingface.co/settings/tokens

### Step 3: Create Project Directory
Ensure the project folder has write permissions for temporary PDF files.

## Running the Application

### Step 1: Navigate to Project Directory
```bash
cd /path/to/RAG_Q&A_Conversation
```

### Step 2: Run the Streamlit Application
```bash
streamlit run app.py
```

### Step 3: Access the Web Interface
The application will open automatically at `http://localhost:8501`

### Step 4: Using the Application
1. **Enter Groq API Key**: Paste your API key in the password field (required to proceed)
2. **Set Session ID**: Enter a unique session ID (defaults to "default_session")
   - Use different IDs for different conversations
   - Same ID continues previous conversations
3. **Upload PDFs**: Click "Choose A PDF file" and select one or multiple PDFs
   - Supported format: PDF only
   - Can upload multiple files at once
4. **Wait for Processing**: The system will:
   - Extract text from PDFs
   - Split documents into chunks
   - Create embeddings
   - Build vector store (this may take a moment)
5. **Ask Questions**: Type your question in "Your question:" field
6. **Review Responses**:
   - Assistant answer appears in the main area
   - Chat history is displayed showing all messages
7. **Continue Conversation**: Ask follow-up questions
   - The system automatically maintains context
   - Previous questions and answers inform current responses

## File Structure
```
RAG_Q&A_Conversation/
├── app.py                # Main Streamlit application
├── readme.md            # This file
└── temp.pdf            # Temporary file storage (auto-generated)
```

## Architecture Diagram
```
User Input
    ↓
Session Management (Session ID)
    ↓
PDF Upload → PDF Processing → Text Splitting
    ↓
Embeddings (HuggingFace) → Vector Store (Chroma)
    ↓
Chat History Management
    ↓
Question Reformulation (History-Aware Retriever)
    ↓
Semantic Search (Retriever)
    ↓
Context + Chat History + Question
    ↓
LLM (Groq Llama 3.1) → Answer Generation
    ↓
Response Display + Chat History Display
```

## Key Technologies

### LangChain Components
- **create_history_aware_retriever**: Reformulates questions based on chat history
- **create_retrieval_chain**: Combines retriever with document chain
- **create_stuff_documents_chain**: Stuffs retrieved documents into prompt
- **RunnableWithMessageHistory**: Manages message history automatically
- **ChatPromptTemplate**: Structures prompts with placeholders
- **MessagesPlaceholder**: Allows dynamic insertion of chat history

### Embeddings & Vector Store
- **HuggingFaceEmbeddings (all-MiniLM-L6-v2)**: Lightweight, efficient embeddings
- **Chroma**: Vector database for storing and retrieving embeddings
- **RecursiveCharacterTextSplitter**: Smart document chunking

### Language Model
- **ChatGroq (Llama 3.1-8B-Instant)**: Fast, efficient language model

### Document Processing
- **PyPDFLoader**: Extracts text from PDF files

## Customization Options

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
```
- Smaller chunks: More specific but more retrieval calls
- Larger chunks: More context but potentially less precise

### Change Embedding Model
```python
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

### Modify Response Length
Edit the system_prompt to change "three sentences maximum" to desired length

### Use Different LLM
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

## Performance Optimization

### Tips for Better Performance
1. **Batch Process**: Upload related PDFs together for coherent context
2. **Session Management**: Use different session IDs for different topics
3. **Chunk Optimization**: Adjust chunk size based on document complexity
4. **Query Specificity**: Ask specific questions for better retrieval

### Memory Management
- Chat history is stored in Streamlit session state
- Temporary PDF files are created in the project directory
- Consider clearing session history for long conversations

## Troubleshooting

### "Please enter the GRoq API Key" warning
- Ensure you've entered a valid Groq API key
- Check that the key has not expired
- Verify correct copy-paste without extra spaces

### PDFs not being processed
- Ensure PDF files are valid and not corrupted
- Check file format is PDF (not images saved as PDFs)
- Verify sufficient disk space for temporary files

### Slow response time
- Reduce `chunk_size` for faster retrieval
- Check internet connection for API calls
- Reduce number of PDFs or use smaller documents
- Consider reducing overlap between chunks

### Chat history not persisting
- Ensure you're using the same session ID
- Check browser cache/cookies are enabled
- Verify Streamlit is not being run in incognito mode

### Memory issues with large PDFs
- Process fewer PDFs at once
- Split large documents before uploading
- Increase system RAM allocation

## Use Cases
- Research paper analysis with follow-up questions
- Technical documentation Q&A with conversation context
- Contract review with multi-turn questioning
- Academic paper discussion
- Knowledge base exploration

## API Costs
- **Groq API**: Currently free tier available (check current pricing)
- **HuggingFace Embeddings**: Local, no additional costs
- **No OpenAI costs** for embeddings in this implementation

## Limitations & Future Improvements
- **Current**: Limited to text extraction from PDFs
- **Future**: Support for images, tables, scanned documents
- **Current**: Linear chat history
- **Future**: Conversation branching, export chat history

## Related Projects
- RAG Document Q&A (single turn)
- Basic Chatbot with Ollama
- Enhanced Chatbot with Ollama
- Chatbot with OpenAI

---

**Last Updated**: January 25, 2026

## Support
For issues or questions:
1. Check the Troubleshooting section
2. Verify API keys and environment variables
3. Ensure all dependencies are installed correctly
4. Check Streamlit and LangChain documentation
