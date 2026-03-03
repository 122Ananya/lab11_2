import os
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document

load_dotenv()

def build_rag(file_path):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

# PDF
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)

#CSV
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        documents = []
        for _, row in df.iterrows():
            text = f"""
            Passenger {row['PassengerId']}:
            Name: {row['Name']}
            Age: {row['Age']}
            Sex: {row['Sex']}
            Class: {row['Pclass']}
            Survived: {row['Survived']}
            """
            documents.append(Document(page_content=text))

        chunks = splitter.split_documents(documents)

    else:
        raise ValueError("Unsupported file type")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain