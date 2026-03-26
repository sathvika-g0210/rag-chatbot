import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_documents(pdf_path: str, index_name: str = "rag-chatbot"):
    print(f"--- Loading: {pdf_path} ---")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    print("--- Uploaded to Pinecone! ---")

if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name="rag-chatbot")
    print("--- FINAL SUCCESS ---")