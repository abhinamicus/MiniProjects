import os
import time
import pickle
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

try:
    # --- Load environment variables ---
    load_dotenv()

    # --- Azure OpenAI Chat config ---
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = "2023-05-15"
    openai.api_key = os.getenv("AZURE_OPENAI_GPT_KEY")
    gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")

    # --- Azure OpenAI Embedding config ---
    embedding_azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    embedding_azure_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

    # --- Load FAISS index and chunks ---
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=embedding_azure_endpoint,
        deployment=deployment,
        model=model,
        api_key=embedding_azure_key,
        api_version="2023-05-15"
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # --- Streamlit UI ---
    st.title("Azure PDF Q&A Bot")

    user_question = st.text_input("Ask a question about Radiohead")

    if user_question:
        # Retrieve top 5 relevant chunks from FAISS
        results = db.similarity_search(user_question, k=5)
        context = "\n\n".join([doc.page_content for doc in results])

        # Compose the prompt for GPT
        system_prompt = (
            "You are an assistant that answers questions based on the provided context. "
            "If the answer is not in the context, say you don't know."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            "Answer:"
        )

        # Call Azure OpenAI GPT
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