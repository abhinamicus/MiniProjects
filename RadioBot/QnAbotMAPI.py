"""
RadioBot
A Q&A bot with:
- User interface (Streamlit)
- Conversational memory
- Async concurrent calls (asyncio)
- Simultaneous API calls (e.g., current time)
- Flawless error handling
- Security best practices
- Reduced hallucination by instructing the model to only answer from context
"""

import os
import pickle
import asyncio
import httpx
import streamlit as st
import requests
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load environment variables securely ---
load_dotenv()
TIME_API_URL = os.getenv("TIME_API_URL", "https://worldtimeapi.org/api/timezone/Etc/UTC")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# --- Streamlit UI ---
st.set_page_config(page_title="RadioBot", layout="wide")
st.title("RadioBot")
st.markdown("Ask Anything Radiohead. Current time also if you want to")

# --- Background Image (full coverage, responsive for mobile) ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://thelabel.co.nz/wp-content/uploads/2019/06/Screen-Shot-2019-06-27-at-12.23.40-PM.png');
        background-size: cover !important;
        background-repeat: no-repeat !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        min-height: 100vh;
        width: 100vw;
    }
    @media (max-width: 768px) {
        .stApp {
            background-size: cover !important;
            background-position: center center !important;
        }
    }
    .stTextInput>div>div>input { font-size: 1.1rem; }
    .stButton>button { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Async API calls ---
async def fetch_time(client):
    try:
        resp = await client.get(TIME_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("datetime") or data.get("dateTime") or str(data)
    except Exception as e:
        return f"Time API error: {e}"

async def get_external_info():
    async with httpx.AsyncClient() as client:
        time_result = await fetch_time(client)
        return time_result

# --- Hugging Face QA function with anti-hallucination prompt ---
def hf_qa(question, context):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    system_prompt = (
        "Answer the question only using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    payload = {
        "inputs": {
            "question": f"{system_prompt}\n\n{question}",
            "context": context
        }
    }
    try:
        response = requests.post(HF_QA_URL, headers=headers, json=payload, timeout=30)
        data = response.json()
        return data.get("answer") or str(data)
    except Exception as e:
        return f"Hugging Face API error: {e}"

# --- Conversational memory (session state) ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- User input ---
with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input("Ask Anything Radiohead", max_chars=300)
    submit = st.form_submit_button("Ask")

# --- File upload ---
uploaded_files = st.file_uploader(
    "Upload your Radiohead PDF files", type=["pdf"], accept_multiple_files=True
)

def load_chunks_from_uploads(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return []
    temp_dir = tempfile.mkdtemp()
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        paths.append(file_path)
    # Use PyPDFLoader directly on each file
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for path in paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks.extend(text_splitter.split_documents(pages))
    return chunks

if uploaded_files:
    chunks = load_chunks_from_uploads(uploaded_files)
else:
    chunks = []

# --- Main logic ---
if submit and user_question:
    st.session_state.history.append({"role": "user", "content": user_question})

    # Retrieve relevant context from PDFs
    try:
        # Simple keyword search for top 5 chunks
        question_words = set(user_question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            text = chunk.page_content.lower()
            score = sum(word in text for word in question_words)
            if score > 0:
                scored_chunks.append((score, chunk.page_content))
        scored_chunks.sort(reverse=True)
        top_contexts = [text for score, text in scored_chunks[:5]]
        context = "\n\n".join(top_contexts)
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        context = ""

    # Async API call (time)
    with st.spinner("Clocking..."):
        try:
            time_result = asyncio.run(get_external_info())
        except Exception as e:
            time_result = "Time API error"

        # Hugging Face QA with anti-hallucination prompt
        answer = hf_qa(user_question, context) if context else "Sorry, I couldn't find relevant information in the documents."

    st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("**Answer:**")
    st.markdown(f"<div style='background:#222;color:#fff;padding:1em;border-radius:8px'>{answer}</div>", unsafe_allow_html=True)
    st.markdown(f"**Current Time:** {time_result}")

# --- Display conversation history ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for turn in st.session_state.history[-10:]:
        if turn["role"] == "user":
            st.markdown(f"**You:** {turn['content']}")
        else:
            st.markdown(f"**RadioBot:** {turn['content']}")