Azure OpenAI PDF Q&A Bot
========================

This project provides a Q&A bot that can answer questions about the content of PDF documents using Azure OpenAI for both embeddings and chat completions. It supports both a command-line interface (Q&Abot.py) and a web interface (app.py using Streamlit).

Features:
---------
- Loads and splits PDF documents into chunks.
- Generates embeddings using Azure OpenAI.
- Stores and searches embeddings locally using FAISS for fast retrieval.
- Uses Azure OpenAI GPT models to answer questions based on retrieved context.
- Supports both CLI and web (Streamlit) interfaces.
- Batches embedding requests to avoid Azure rate limits.
- Loads/saves all data for fast reruns.

Setup:
------
1. Clone this repository to your local machine.
2. Install dependencies:
   pip install -r requirements.txt

3. Copy `example.env` to `.env` and fill in your Azure OpenAI credentials and deployment names.

4. Place your PDF files in the directory specified by `PDF_DIRECTORY_PATH` in your `.env`.

Usage:
------
- **Command-line bot:**  
  Run `python Q&Abot.py`  
  Type your question at the prompt. Type `exit` or `quit` to stop.

- **Web app (Streamlit):**  
  Run `streamlit run app.py`  
  Open the provided URL in your browser and ask questions in the web interface.

Security:
---------
- Never commit your real `.env` file or Azure keys to public repositories.
- FAISS index loading uses `allow_dangerous_deserialization=True` because you are loading your own trusted pickle files. Never load untrusted pickle files.

Deployment:
-----------
- For sharing with non-technical users, deploy the Streamlit app to [Streamlit Community Cloud](https://streamlit.io/cloud) or Azure App Service.

Files:
------
- Q&Abot.py: Command-line Q&A bot.
- app.py: Streamlit web app for Q&A.
- example.env: Example environment variable file (copy to `.env` and fill in).
- requirements.txt: Python dependencies.

License:
--------
MIT