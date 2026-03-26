import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def create_chatbot(index_name: str = "rag-chatbot"):
    # 1. Connect to Pinecone
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Setup the Brain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below:
    {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask(chain, question: str) -> str:
    return chain.invoke(question)

# For testing directly in terminal
if __name__ == "__main__":
    chain = create_chatbot()
    print("\n--- 🤖 RESUME BOT IS ONLINE ---")
    print("Ask a question about the candidate (type 'exit' to stop):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']: break
        result = ask(chain, query)
        print(f"\nAI: {result}")