RadioBot - PDF Q&A Streamlit App
================================

RadioBot is a Streamlit web application that allows you to ask questions about a collection of Radiohead-related PDF documents.
It uses Azure OpenAI for both document embeddings (semantic search) and GPT-based answers.

-------------------------------------------------------------------------------

Project Structure:
------------------
app.py             - Main Streamlit app. Handles PDF loading, embedding, search, and UI.
requirements.txt   - Python dependencies for the project.
.env               - Your Azure OpenAI credentials (not tracked by git).
example.env        - Template for .env. Copy and fill with your own keys.
pdfs/              - Folder containing all PDF files to be indexed and searched.
chunks.pkl         - Cached document chunks (auto-generated for faster startup).
faiss_index        - Cached FAISS vector index (auto-generated for faster search).
.gitignore         - Ensures sensitive files like .env are not tracked by git.

-------------------------------------------------------------------------------

How It Works:
-------------
1. All PDF files in the 'pdfs/' folder are loaded and split into text chunks.
2. Each chunk is embedded using Azure OpenAI Embeddings and stored in a FAISS vector index.
3. When you ask a question, the app finds the most relevant chunks using semantic search.
4. The selected context is sent to Azure OpenAI GPT, which generates an answer.
5. The answer is displayed in a styled box in the Streamlit web interface.

-------------------------------------------------------------------------------

Setup Instructions:
-------------------
1. **Clone the repository:**
   git clone <your-repo-url>
   cd QnA_Bot

2. **Create and activate a virtual environment (recommended):**
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Set up environment variables:**
   - Copy 'example.env' to '.env'
   - Fill in your Azure OpenAI credentials:
     AZURE_OPENAI_ENDPOINT=...
     AZURE_OPENAI_GPT_KEY=...
     AZURE_OPENAI_GPT_DEPLOYMENT=...
     AZURE_OPENAI_EMBEDDING_ENDPOINT=...
     AZURE_OPENAI_EMBEDDING_KEY=...
     AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
     AZURE_OPENAI_EMBEDDING_MODEL=...

5. **Add your PDF files:**
   - Place all relevant PDFs in the 'pdfs/' folder.

6. **Run the app:**
   streamlit run app.py

7. **Access the app:**
   - Open the provided local URL in your browser (e.g., http://localhost:8501).

-------------------------------------------------------------------------------

Usage:
------
- Enter your question in the input box and click the button ("Come on!").
- The app will search the PDFs for relevant information and use GPT to answer your question.
- Answers are displayed in a dark box for readability.

-------------------------------------------------------------------------------

Notes:
------
- The first run may take longer as embeddings are generated and cached.
- If you hit Azure OpenAI rate limits, the app will automatically retry.
- Some UI elements (like centering and form hints) are controlled by Streamlit and may not be fully customizable.
- The background image is set via CSS and may not display perfectly on all devices.

-------------------------------------------------------------------------------

Troubleshooting:
----------------
- **.env not found:** Make sure you have copied and