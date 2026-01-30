# Text Summarization from YouTube Videos

## Project Overview
This Streamlit application extracts subtitles from YouTube videos and generates concise ~300-word summaries using Groq's `llama-3.1-8b-instant` model via LangChain. The app uses `yt-dlp` for reliable subtitle extraction and LangChain's `map_reduce` chain for intelligent summarization.

## Features
- **YouTube Subtitle Extraction**: Uses `yt-dlp` to download subtitles in VTT format (more reliable than API-based approaches)
- **Automatic VTT Parsing**: Converts VTT subtitle format to plain text
- **Fast Summarization**: Leverages Groq's optimized Llama 3.1 model for quick responses
- **LangChain Map-Reduce Chain**: Splits large documents and summarizes them efficiently
- **Simple Streamlit UI**: User-friendly interface with URL input and one-click summarization
- **Error Handling**: Comprehensive error messages for missing subtitles or invalid URLs
- **Temp Directory Management**: Uses Python's `tempfile` for safe, automatic cleanup

## Step-by-Step Implementation Details

### 1. **Imports and Dependencies**
```python
import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os, subprocess, tempfile
from langchain.schema import Document
```

- **`validators`**: URL format validation
- **`streamlit`**: Web UI framework
- **`langchain`**: LLM orchestration and summarization chains
- **`langchain_groq`**: Groq API integration
- **`dotenv`**: Environment variable loading
- **`subprocess`**: Execute `yt-dlp` command for subtitle download
- **`tempfile`**: Create temporary directories for subtitle files
- **`Document`**: LangChain document wrapper for text

### 2. **Environment Setup**
```python
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```
- Loads `GROQ_API_KEY` from `.env` file
- Required for Groq API authentication

### 3. **Streamlit Page Configuration**
```python
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')
```
- Sets browser tab title and icon
- Displays main title and subtitle

### 4. **YouTube Subtitle Extraction Function** (Most Important)
```python
def load_youtube_transcript(url: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--skip-download",              # Don't download video file
            "--write-auto-sub",             # Include auto-generated subtitles
            "--write-sub",                  # Include manual subtitles
            "--sub-lang", "en",             # English subtitles only
            "--sub-format", "vtt",          # VTT format (text-based)
            "-o", f"{tmpdir}/%(id)s.%(ext)s",
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError("Failed to fetch subtitles using yt-dlp")
```

**Key Steps:**
1. Creates temporary directory for subtitle files
2. Constructs `yt-dlp` command with options to download subtitles in VTT format
3. Executes command via `subprocess.run()`
4. Returns error if download fails

**VTT Format Parsing:**
```python
        # Find subtitle file
        subtitle_file = None
        for file in os.listdir(tmpdir):
            if file.endswith(".vtt"):
                subtitle_file = os.path.join(tmpdir, file)
                break
        
        # Parse VTT → plain text
        lines = []
        with open(subtitle_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip metadata lines: timestamps, headers, notes
                if line and "-->" not in line and not line.startswith(("WEBVTT", "NOTE")):
                    lines.append(line)
        
        text = " ".join(lines)
        return [Document(page_content=text)]
```

**Parsing Details:**
- Finds the generated `.vtt` file in temp directory
- Reads VTT file and filters out:
  - Timestamps (lines containing `-->`)
  - VTT header (`WEBVTT`)
  - Notes
- Joins remaining lines into single text string
- Returns LangChain `Document` object for chain compatibility

### 5. **User Input**
```python
generic_url = st.text_input("URL", label_visibility="collapsed")
```
- Text input field for YouTube URL
- Label is hidden; placeholder is the input field itself

### 6. **LLM Configuration**
```python
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
```
- Initializes Groq Llama 3.1 model
- Uses API key from environment

### 7. **Summarization Prompt Template**
```python
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
```
- Defines prompt structure with placeholder `{text}`
- Instructs model to generate 300-word summary

### 8. **Summarization Button and Workflow**
```python
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    
    else:
        try:
            with st.spinner("Waiting..."):
                # Extract subtitles
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)
                else:
                    st.error("Please enter a valid YouTube URL")
                
                # Create summarization chain
                chain = load_summarize_chain(
                    llm, 
                    chain_type="map_reduce",
                    map_prompt=prompt,
                    combine_prompt=prompt
                )
                
                # Run summarization
                output_summary = chain.run(docs)
                st.success(output_summary)
        
        except Exception as e:
            st.exception(e)
```

**Workflow:**
1. **Validation**: Check API key and URL are provided and URL is valid format
2. **Subtitle Extraction**: Call `load_youtube_transcript()` to extract and parse subtitles
3. **Map-Reduce Chain**: 
   - `map_prompt`: Summarizes each document chunk
   - `combine_prompt`: Combines chunk summaries into final summary
4. **Display**: Shows success message with summary or error if something fails

### 9. **Map-Reduce Summarization Chain Explained**
The `chain_type="map_reduce"` is optimized for large documents:
- **Map Phase**: Splits large subtitle text into chunks, summarizes each independently
- **Reduce Phase**: Combines all chunk summaries into a single coherent summary
- Both phases use the same prompt template

## How to Execute the App

### Step 1: Install Dependencies
```bash
pip install streamlit validators langchain langchain-groq python-dotenv yt-dlp
```

**Important Notes:**
- `yt-dlp` requires FFmpeg to be installed on your system
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: Download from ffmpeg.org or use `choco install ffmpeg`

### Step 2: Create `.env` File
Create a file named `.env` in the project directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com

### Step 3: Run the App
```bash
cd /Users/deepti/Documents/git/generative-ai/text_summarization_video
streamlit run app.py
```

### Step 4: Access in Browser
- Streamlit will automatically open `http://localhost:8501` in your default browser
- If not, manually navigate to that URL

### Step 5: Use the App
1. Paste a YouTube URL in the input field
2. Click "Summarize the Content from YT or Website"
3. Wait for the spinner (subtitle extraction + summarization takes 5-15 seconds)
4. View the generated summary

## Why `yt-dlp` Instead of `youtube_transcript_api`?

| Feature | `yt-dlp` | `youtube_transcript_api` |
|---------|----------|--------------------------|
| **Reliability** | Very reliable | Sometimes blocked by YouTube |
| **Subtitle Types** | Manual + auto-generated | Manual only |
| **Authentication** | No API key needed | Can be blocked |
| **Fallback** | Auto-generated available | None |
| **Speed** | Fast | Varies |


**Last Updated**: January 29, 2026
