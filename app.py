import streamlit as st
from rag_pipeline_1RVU23CSE051 import build_rag

# st.set_page_config(page_title="Chat with PDF", layout="centered")

st.title("Ask information from the PDF(Ananya_051 - Lab 11.2)")

pdf_file = st.selectbox(
    "Select document",
    ["data/invoice1.pdf", "data/invoice2.pdf", "data/titanic.csv"]
)

if "qa" not in st.session_state :
    st.session_state.qa = build_rag(pdf_file) 

question = st.text_input("Ask a question?")

if st.button("SEND"):
    response = st.session_state.qa.invoke({"query": question})

    st.subheader("ANSWER")
    st.write(response["result"])

    st.subheader("RETRIEVED CHUNKS")
    for doc in response["source_documents"]:
        st.write(doc.page_content)
        
