import streamlit as st
from dotenv import load_dotenv
from document_loader import extract_text
from qa_engine import build_vector_store, answer_query

load_dotenv()

st.set_page_config(page_title="Gemini DocQA")
st.title("📄🤖 Gemini-Powered DocQA")
st.markdown("Upload a document (PDF/DOCX) and ask questions!")

uploaded_file = st.file_uploader("📁 Upload your file", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    with st.spinner("Processing..."):
        docs = build_vector_store(text)

    query = st.text_input("🔍 Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            answer = answer_query(docs, query)
            st.write("**Answer:**", answer)
