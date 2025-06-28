import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def build_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return docs  # Gemini doesn't require vector store

def answer_query(docs, query):
    context = "\n".join([doc.page_content for doc in docs[:5]])

    prompt = f"You are an intelligent assistant. Use the following document to answer the question.\n\nDocument:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
