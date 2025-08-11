import os
import time
import pickle
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# --- Load Environment Variables ---
load_dotenv()

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
# openai.api_base = os.getenv("AZURE_OPENAI_GPT_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_GPT_KEY")
gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")

embedding_azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
embedding_azure_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
pdf_dir_path = os.getenv("PDF_DIRECTORY_PATH")

print("Model:", model)

# --- Initialize Azure OpenAI Embeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=embedding_azure_endpoint,
    deployment=deployment,
    model=model,
    api_key=embedding_azure_key,
    api_version="2023-05-15"
)

# --- Try to load embeddings and chunks from disk ---
if os.path.exists("embeddings.pkl") and os.path.exists("chunks.pkl"):
    print("🔄 Loading precomputed embeddings and chunks from disk...")
    with open("embeddings.pkl", "rb") as f:
        embeddings_list = pickle.load(f)
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
else:
    # --- Load all PDFs from directory ---
    if not os.path.exists(pdf_dir_path):
        raise FileNotFoundError(f"❌ Directory not found at: {pdf_dir_path}")

    loader = DirectoryLoader(
        path=pdf_dir_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pages = loader.load()

    # --- Chunk PDF Content ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # --- Generate Embeddings in batches ---
    print(f"🔁 Generating embeddings for {len(chunks)} chunks...")
    batch_size = 5  # Adjust as needed for your rate limit
    embeddings_list = []
    for i in range(0, len(chunks), batch_size):
        batch = [chunk.page_content for chunk in chunks[i:i+batch_size]]
        embeddings_list.extend(embeddings.embed_documents(batch))
        print(f"Processed batch {i//batch_size + 1}")
        time.sleep(10)  # Wait 10 seconds between batches (adjust as needed)

    print(f"✅ Done! Generated {len(embeddings_list)} embeddings.")
    print(f"🔹 Sample vector (first 5 dims): {embeddings_list[0][:5]}")

    # --- Save embeddings and chunks to disk ---
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_list, f)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Embeddings and chunks saved to embeddings.pkl and chunks.pkl")

# --- Create or load FAISS vector store ---
if os.path.exists("faiss_index"):
    print("🔄 Loading FAISS index from disk...")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⚡ Creating FAISS index...")
    db = FAISS.from_embeddings(
        list(zip([chunk.page_content for chunk in chunks], embeddings_list)),
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks]
    )
    db.save_local("faiss_index")

def ask_question_loop():
    print("\n🔎 Q&A Bot is ready! Type your question (or 'exit' to quit):")
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

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

        # Call Azure OpenAI GPT (new API style for openai>=1.0.0)
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
        print("\nBot:", answer)
        print("-" * 60)

ask_question_loop()