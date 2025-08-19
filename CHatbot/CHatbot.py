import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import base64
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

# --- Set background image ---
def set_bg(image_file):
    with open(image_file, "rb") as img:
        img_bytes = img.read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            width: 100vw;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Error handling wrapper ---
def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {e}")
            st.write(e)
    return wrapper

def load_and_create_db():
    # Set your PDF directory path here
    pdf_dir = r"B:\Python\pdfs"  # Change to your actual PDF directory

    docs = []
    if os.path.isdir(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        # Create embeddings and vector database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(docs, embeddings)
        vectordb.persist()
        return vectordb
    else:
        raise ValueError("PDF directory does not exist.")

@safe_run
def main():
    st.title("RadioBot")
    set_bg("Thom.png")  # Place your image in the same folder

    # Load or create the vector database
    if os.path.exists("db"):
        vectordb = Chroma(persist_directory="db", embedding_function=HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2"))
    else:
        vectordb = load_and_create_db()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

    chat_llm = AzureChatOpenAI(
        openai_api_version="2023-05-15",  # Use your Azure OpenAI API version
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=512,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    # Display chat history
    for user_msg, bot_msg in st.session_state["history"]:
        st.markdown(
            f"""
            <div style="background-color: rgba(240,240,240,0.85); padding: 0.5em 1em; border-radius: 8px; margin-bottom: 0.5em;">
                <b>You:</b> {user_msg}
            </div>
            <div style="background-color: rgba(255,255,255,0.95); padding: 0.5em 1em; border-radius: 8px; margin-bottom: 1em;">
                <b>RadioBot:</b> {bot_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.form("chat_form"):
        user_input = st.text_input(
            label="",
            placeholder="Ask Anything Radiohead",
            key="user_input"
        )
        submitted = st.form_submit_button("Come on!")
    if submitted and user_input:
        try:
            response = chain.invoke({"question": user_input})
            st.session_state["history"].append((user_input, response["answer"]))
            st.markdown(
                f"""
                <div style="background-color: rgba(255,255,255,0.85); padding: 1em; border-radius: 8px; margin-top: 1em;">
                    {response["answer"]}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.session_state["user_input"] = ""  # Clear the input box
        except Exception as e:
            st.error(f"Chat error: {e}")
            st.write(e)

if __name__ == "__main__":
    main()