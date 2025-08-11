import os
import pickle
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

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
    pdf_dir_path = os.getenv("PDF_DIRECTORY_PATH", "pdfs")  # default to 'pdfs' folder

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
            embeddings_list.extend(embeddings.embed_documents(batch))
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        db = FAISS.from_embeddings(
            list(zip([chunk.page_content for chunk in chunks], embeddings_list)),
            embedding=embeddings,
            metadatas=[chunk.metadata for chunk in chunks]
        )
        db.save_local("faiss_index")

    # --- Streamlit UI ---
    st.title("Azure PDF Q&A Bot")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
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
        st.write(answer)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()