import streamlit as st
import tempfile
import os
from src.chat import create_chatbot, ask
from src.ingest import ingest_documents

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("Ask My Documents")
st.caption("Powered by LangChain · OpenAI · Pinecone")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and uploading your document... please wait!"):
        ingest_documents(tmp_path)
    
    os.unlink(tmp_path)
    st.success("Document uploaded! You can now ask questions about it.")

# Load chatbot
@st.cache_resource
def load_chain():
    return create_chatbot()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask anything about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(chain, question)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})