import os
import time
import pickle
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://thelabel.co.nz/wp-content/uploads/2019/06/Screen-Shot-2019-06-27-at-12.23.40-PM.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    /* Force the main container to the left */
    .main .block-container {
        margin-left: 0 !important;
        margin-right: auto !important;
        max-width: 600px !important;
        padding-left: 2rem !important;
        background: rgba(255,255,255,0.7);
        border-radius: 10px;
        box-shadow: 2px 2px 16px rgba(0,0,0,0.08);
        position: relative;
        left: 0 !important;
        display: block !important;
    }
    /* Aggressively left-align all forms and their parents */
    section[data-testid="stSidebar"], .stForm, .stForm form, .stForm .stTextInputContainer, .stForm input, .stForm button {
        align-items: flex-start !important;
        justify-content: flex-start !important;
        text-align: left !important;
        margin-left: 0 !important;
        width: 100% !important;
        display: block !important;
    }
    /* Hide 'Press Enter to submit form' in all possible locations */
    .stForm .stMarkdown, .stForm label[data-testid="stMarkdownContainer"], .stForm div[role="alert"] {
        display: none !important;
    }
    /* Hide 'Press Enter to submit form' in all possible locations */
    .stForm .stMarkdown, .stForm label[data-testid="stMarkdownContainer"], .stForm div[role="alert"] {
        display: none !important;
    }
    .answer-box {
        background: rgba(30, 30, 30, 0.85);
        color: #fff !important;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 500;
        word-break: break-word;    
    }
    </style>
    """,
    unsafe_allow_html=True
)    

try:
    load_dotenv()

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = "2023-05-15"
    openai.api_key = os.getenv("AZURE_OPENAI_GPT_KEY")
    gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")

    embedding_azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    embedding_azure_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    pdf_dir_path = os.path.join(os.path.dirname(__file__), "pdfs")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=embedding_azure_endpoint,
        deployment=deployment,
        model=model,
        api_key=embedding_azure_key,
        api_version="2023-05-15"
    )

    # --- Check for precomputed files ---
    if os.path.exists("chunks.pkl") and os.path.exists("faiss_index"):
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # --- Load PDFs and process ---
        loader = DirectoryLoader(
            path=pdf_dir_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        # --- Generate embeddings ---
        st.info("Generating embeddings, this may take a while...")
        batch_size = 5
        embeddings_list = []
        for i in range(0, len(chunks), batch_size):
            batch = [chunk.page_content for chunk in chunks[i:i+batch_size]]
            success = False
            while not success:
                try:
                    embeddings_list.extend(embeddings.embed_documents(batch))
                    success = True
                except Exception as e:
                    if "429" in str(e):
                        st.warning("Rate limit hit. Waiting 5 seconds before retrying...")
                        time.sleep(5)
                    else:
                        raise e
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        db = FAISS.from_embeddings(
            list(zip([chunk.page_content for chunk in chunks], embeddings_list)),
            embedding=embeddings,
            metadatas=[chunk.metadata for chunk in chunks]
        )
        db.save_local("faiss_index")

    # --- Streamlit UI ---
    
    st.title("RadioBot")

    with st.form("question_form"):
        user_question = st.text_input("Ask something Radiohead")
        submitted = st.form_submit_button("Come on!")

    if submitted and user_question:
        results = db.similarity_search(user_question, k=5)
        context = "\n\n".join([doc.page_content for doc in results])
        system_prompt = (
            "You are an assistant that answers questions based on the provided context. "
            "If the answer is not in the context, say you don't know."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            "Answer:"
        )
        with st.spinner("Thinking..."):
            response = openai.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )
            answer = response.choices[0].message.content
        st.markdown("**Answer:**")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()