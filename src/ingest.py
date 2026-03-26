import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# This line reads your API keys from your .env file
load_dotenv()

def ingest_docs():
    print("--- Starting Document Ingestion ---")
    
    # 1. Load the PDF from your data folder
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    if not docs:
        print("Error: No PDF files found in the 'data' folder.")
        return
        
    print(f"Success: Loaded {len(docs)} pages.")

    # 2. Split the text into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Success: Split document into {len(chunks)} chunks.")

    # 3. Create Embeddings and upload to Pinecone
    embeddings = OpenAIEmbeddings()
    index_name = "rag-chatbot" 
    
    print("Uploading to Pinecone... this might take a moment.")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name
    )
    print("--- FINAL SUCCESS: Your resume is now in the database! ---")

if __name__ == "__main__":
    ingest_docs()