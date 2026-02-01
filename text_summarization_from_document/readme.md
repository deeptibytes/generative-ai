# Text Summarization from PDF Documents

## Project Overview
This Jupyter notebook demonstrates three different approaches to document summarization using LangChain and Groq's Llama 3.1 model. It loads a PDF document (APJ Abdul Kalam's speech) and applies Stuff, Map-Reduce, and Refine summarization chains to generate summaries of varying quality and efficiency.

## Features
- **PDF Document Loading**: Uses PyPDFLoader to extract text from PDF files
- **Multiple Summarization Strategies**: Demonstrates Stuff, Map-Reduce, and Refine chains
- **Custom Prompts**: Tailored prompts for different summarization approaches
- **Document Chunking**: Splits large documents for efficient processing
- **Verbose Output**: Shows intermediate steps for learning purposes
- **Groq Integration**: Uses Llama 3.1 model for fast, high-quality summarization

## Step-by-Step Implementation Details

### 1. **Environment Setup**
```python
import os
from dotenv import load_dotenv
load_dotenv()
```
- Loads environment variables from `.env` file

### 2. **LLM Configuration**
```python
from langchain_groq import ChatGroq
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant")
```
- Initializes Groq's Llama 3.1 model for summarization

### 3. **Stuff Chain Summarization**
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("apjspeech.pdf")
docs = loader.load_and_split()

from langchain import PromptTemplate
template = """ Write a concise and short summary of the following speech, Speech :{text} """
prompt = PromptTemplate(input_variables=['text'], template=template)

from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=True)
output_summary = chain.invoke(docs)
```
- Loads and splits the PDF into documents
- Uses Stuff chain to summarize all documents in one prompt
- Custom prompt for concise summary

### 4. **Map-Reduce Chain Preparation**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
final_documents = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(docs)

chunks_prompt = """
Please summarize the below speech:
Speech:`{text}'
Summary:
"""
map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

final_prompt = '''
Provide the final summary of the entire speech with these important points.
Add a Motivation Title,Start the precise summary with an introduction and provide the summary in number points for the speech.
Speech:{text}
'''
final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)
```
- Splits documents into 2000-character chunks with 100-character overlap
- Defines map prompt for individual chunk summarization
- Defines combine prompt for final summary with structured format

### 5. **Map-Reduce Chain Execution**
```python
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
)
output = summary_chain.invoke(final_documents)
```
- Runs map-reduce summarization with custom prompts

### 6. **Refine Chain Summarization**
```python
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    verbose=True
)
output_summary = chain.run(final_documents)
```
- Uses default refine prompts to iteratively build summary

## Summarization Chain Differences

### 1. **Stuff Chain**
**How it works:**
- Combines all documents into a single prompt
- Sends everything to the LLM at once
- Uses a single API call

**Pros:**
- Simple and fast for small documents
- Preserves all context in one go
- No intermediate processing

**Cons:**
- Limited by LLM's context window (typically 4K-32K tokens)
- Not suitable for large documents
- Single point of failure

**Best for:** Small documents, quick summaries, when context preservation is critical

### 2. **Map-Reduce Chain**
**How it works:**
- **Map Phase**: Splits documents into chunks, summarizes each chunk independently
- **Reduce Phase**: Combines all chunk summaries into a final coherent summary
- Uses multiple API calls (one per chunk + one for combination)

**Pros:**
- Handles large documents by parallel processing
- Scalable to any document size
- Customizable map and combine prompts for different summary styles

**Cons:**
- More complex setup
- Multiple API calls increase cost and latency
- May lose some cross-chunk context

**Best for:** Large documents, when you need structured summaries, parallel processing

### 3. **Refine Chain**
**How it works:**
- Starts with the first document/chunk
- Generates initial summary
- Iteratively refines the summary by incorporating each subsequent chunk
- Builds understanding progressively

**Pros:**
- Maintains context across the entire document
- Can improve summary quality through iterative refinement
- Good for documents where later parts depend on earlier context

**Cons:**
- Sequential processing (slower than parallel map-reduce)
- Each iteration depends on the previous summary
- May accumulate errors if early summaries are poor

**Best for:** Documents with sequential dependencies, when you want to build comprehensive understanding

## Comparison Table

| Aspect | Stuff Chain | Map-Reduce Chain | Refine Chain |
|--------|-------------|------------------|--------------|
| **Document Size** | Small | Any size | Any size |
| **API Calls** | 1 | Multiple | Multiple |
| **Processing** | Single pass | Parallel then combine | Sequential |
| **Context Preservation** | Excellent | Good (within chunks) | Excellent |
| **Speed** | Fastest | Medium | Slowest |
| **Complexity** | Simple | Medium | Medium |
| **Customization** | Basic | High | Medium |
| **Use Case** | Quick summaries | Large docs, structured output | Context-dependent docs |


Each chain will produce different summary outputs. Compare them to understand the trade-offs.


## Use Cases
- Academic paper summarization
- Meeting transcript analysis
- Legal document review
- Research article synthesis
- Book chapter summaries
- News article aggregation


## Troubleshooting

### PDF Loading Issues
- Ensure PDF is text-based (not scanned images)
- Check file path is correct
- Install PyPDF2 if needed: `pip install PyPDF2`

### API Rate Limiting
- Groq has rate limits; add delays between calls if needed
- Monitor usage at https://console.groq.com

### Memory Issues with Large PDFs
- Increase chunk size for fewer, larger chunks
- Use map-reduce for very large documents
- Consider document preprocessing

### Verbose Output Too Long
- Set `verbose=False` in production
- Run individual cells to isolate output

## Future Enhancements
- Support for multiple document formats (DOCX, HTML)
- Interactive UI for parameter tuning
- Summary quality comparison metrics
- Integration with vector databases for retrieval
- Multi-language summarization

---

**Last Updated**: January 31, 2026
