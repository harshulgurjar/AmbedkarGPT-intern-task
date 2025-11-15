import streamlit as st
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

st.title("📘 AmbedkarGPT – RAG Q&A System")
st.write("Ask any question based on Dr. B.R. Ambedkar's speech.")


def load_docs():
    loader = TextLoader("speech.txt")
    documents = loader.load()
    return documents


def split_docs(documents):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="vector_db"
    )
    return vectordb


def create_qa(vectordb):
    llm = Ollama(model="mistral")
    retriever = vectordb.as_retriever()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

documents = load_docs()
docs = split_docs(documents)
vectordb = create_vector_db(docs)
qa_chain = create_qa(vectordb)


query = st.text_input("Enter your question")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
    
    st.subheader("Answer:")
    st.write(result["result"])
    
    st.subheader("Sources:")
    for doc in result["source_documents"]:
        st.code(doc.page_content)
