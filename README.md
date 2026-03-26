# RAG Chatbot — Ask Questions Over Any Document

A production-style Retrieval-Augmented Generation (RAG) chatbot built with LangChain, OpenAI GPT-3.5, and Pinecone vector database.

## 🔗 Live Demo
[Click here to try it](https://rag-chatbot-ppktmg5ahwxq3zb448p7g.streamlit.app)

## What it does
- Upload **any PDF** document
- Ask questions in natural language
- Get accurate answers grounded in your document (no hallucination)

## Tech Stack
- **LangChain** — orchestration and retrieval chain
- **OpenAI** — embeddings + GPT-3.5-turbo for generation
- **Pinecone** — vector database for semantic search
- **Streamlit** — web frontend
- **Python 3.12**

## Architecture
User uploads PDF → chunked into pieces → OpenAI embeddings → stored in Pinecone → user asks question → semantic search → GPT-3.5 generates grounded answer

## How to run locally
1. Clone this repo
2. Create `.env` file with your API keys
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Author
Sathvika Gourisetty — AI Engineer
[LinkedIn](https://linkedin.com/in/sathvika-gourisetty) | [GitHub](https://github.com/sathvika-g0210)