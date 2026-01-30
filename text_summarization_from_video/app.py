import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os
from langchain.schema import Document
import subprocess
import tempfile
import validators


load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

def load_youtube_transcript(url: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", f"{tmpdir}/%(id)s.%(ext)s",
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError("Failed to fetch subtitles using yt-dlp")

        # Find subtitle file
        subtitle_file = None
        for file in os.listdir(tmpdir):
            if file.endswith(".vtt"):
                subtitle_file = os.path.join(tmpdir, file)
                break

        if not subtitle_file:
            raise RuntimeError("No subtitles available for this video")

        # Parse VTT → plain text
        lines = []
        with open(subtitle_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "-->" not in line and not line.startswith(("WEBVTT", "NOTE")):
                    lines.append(line)

        text = " ".join(lines)

        return [Document(page_content=text)]

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model Using Groq API
llm =ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)

                else:
                    st.error("Please enter a valid YouTube URL")
                

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="map_reduce",map_prompt=prompt, combine_prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(e)
                    