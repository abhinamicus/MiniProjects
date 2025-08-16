"""
RadioBot
A Q&A bot with:
- User interface (Streamlit)
- Conversational memory
- Async concurrent calls (asyncio)
- Simultaneous API calls (e.g., current time, currency rates)
- Flawless error handling
- Security best practices
"""

import os
import pickle
import asyncio
import httpx
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load environment variables securely ---
load_dotenv()
TIME_API_URL = os.getenv("TIME_API_URL", "https://worldtimeapi.org/api/timezone/Etc/UTC")
CURRENCY_API_URL = os.getenv("CURRENCY_API_URL", "https://api.exchangerate.host/latest?base=USD")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# --- Streamlit UI ---
st.set_page_config(page_title="RadioBot", layout="wide")
st.title("RadioBot")
st.markdown("Ask Anything Radiohead. Current time and currency rates also if you want to")

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

# --- PDF Directory ---
pdf_dir = "B:\\Python\\pdfs"  # or your actual folder path

# --- Load or create document chunks ---
@st.cache_resource(show_spinner="Loading Radiohead knowledge base...")
def load_chunks(pdf_dir):
    if os.path.exists("chunks.pkl") and os.path.exists("chunks_dir.txt"):
        with open("chunks_dir.txt", "r") as f:
            cached_dir = f.read().strip()
        if cached_dir == pdf_dir and os.path.exists("chunks.pkl"):
            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            return chunks
    loader = DirectoryLoader(
        path=pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("chunks_dir.txt", "w") as f:
        f.write(pdf_dir)
    return chunks

chunks = load_chunks(pdf_dir)

# --- Async API calls ---
async def fetch_time(client):
    try:
        resp = await client.get(TIME_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("datetime") or data.get("dateTime") or str(data)
    except Exception as e:
        return f"Time API error: {e}"

async def fetch_currency(client):
    try:
        resp = await client.get(CURRENCY_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        usd_to_eur = data.get("rates", {}).get("EUR")
        return f"USD to EUR: {usd_to_eur}" if usd_to_eur else "Currency data unavailable"
    except Exception as e:
        return f"Currency API error: {e}"

async def get_external_info():
    async with httpx.AsyncClient() as client:
        time_task = fetch_time(client)
        currency_task = fetch_currency(client)
        time_result, currency_result = await asyncio.gather(time_task, currency_task)
        return time_result, currency_result

# --- Hugging Face QA function ---
def hf_qa(question, context):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": {"question": question, "context": context}}
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

    # Async API calls (time, currency)
    with st.spinner("Thinking and fetching external info..."):
        try:
            time_result, currency_result = asyncio.run(get_external_info())
        except Exception as e:
            time_result, currency_result = "Time API error", "Currency API error"

        # Hugging Face QA
        answer = hf_qa(user_question, context) if context else "Sorry, I couldn't find relevant information in the documents."

    st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("**Answer:**")
    st.markdown(f"<div style='background:#222;color:#fff;padding:1em;border-radius:8px'>{answer}</div>", unsafe_allow_html=True)
    st.markdown(f"**Current Time:** {time_result}")
    st.markdown(f"**Currency Rate:** {currency_result}")

# --- Display conversation history ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for turn in st.session_state.history[-10:]:
        if turn["role"] == "user":
            st.markdown(f"**You:** {turn['content']}")
        else:
            st.markdown(f"**RadioBot:** {turn['content']}")