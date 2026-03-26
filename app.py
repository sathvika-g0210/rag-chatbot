import streamlit as st
from src.chat import create_chatbot, ask

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("Ask My Documents")
st.caption("Powered by LangChain · OpenAI · Pinecone")

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