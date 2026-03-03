import streamlit as st
from rag_pipeline_1RVU23CSE051 import build_rag

st.set_page_config(page_title="Chat with PDF", layout="centered")

st.title("Chat with Your PDF (Lab 11.2)")

pdf_file = st.selectbox(
    "Select document",
    ["data/invoice1.pdf", "data/invoice2.pdf", "data/titanic.csv"]
)

if "qa" not in st.session_state or st.session_state.get("current_file") != pdf_file:
    st.session_state.qa = build_rag(pdf_file)
    st.session_state.current_file = pdf_file

question = st.text_input("Ask a question from the document")

if st.button("Ask"):
    response = st.session_state.qa(question)

    st.subheader("Answer")
    st.write(response["result"])

    st.subheader("Retrieved Chunks")
    for doc in response["source_documents"]:
        st.write(doc.page_content)