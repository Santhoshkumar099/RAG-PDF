import os
import streamlit as st
import fitz
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
from dotenv import load_dotenv
import time

# Load .env file
load_dotenv()

# Domo API Credentials
DOMO_DEVELOPER_TOKEN = os.getenv("DOMO_DEVELOPER_TOKEN")
API_URL = "https://gwcteq-partner.domo.com/api/ai/v1/text/generation"

# Function to extract text from PDF
def get_pdf_processed(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    return text

# Initialize the vector store
def initialize_vector_store():
    pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        with st.spinner("Loading PDF..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            st.session_state.docs = get_pdf_processed(pdf_files)
            st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
            st.session_state.vector = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
            st.success("PDF content loaded successfully!")

# Function to call the Domo API for responses
def query_domo_api(prompt):
    payload = {
        "input": prompt,  
        "model": "domo.domo_ai.domogpt-chat-medium-v1.1:anthropic",
        "system": """You are a chatbot that answers questions based on the given PDF text. Provide concise answers, limited to 2 lines, ensuring clarity and relevance."""
    }

    headers = {
        "Content-Type": "application/json",
        "X-DOMO-Developer-Token": DOMO_DEVELOPER_TOKEN
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result.get("output", "No response received.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to handle user input
def user_input(prompt):
    # Call Domo API
    start = time.process_time()
    response = query_domo_api(prompt)
    st.write(response)
    st.write("Response time: ", time.process_time() - start)
    
    with st.expander("Expand to view your question in PDF"):
        # Find and display relevant content from the PDF
        found = False
        for page in st.session_state.vector.similarity_search(prompt):
            st.write(page.page_content)
            st.write("-----------------------------")
            found = True
        if not found:
            st.write("No relevant content found in the PDF.")

# Main function for Streamlit interface
def main():
    st.title("Ask Questions from PDF Documents")
    
    # Initialize vector store (PDF processing)
    initialize_vector_store()
    
    # Input from user for a query
    prompt = st.text_input("Input your question here")
    if prompt:
        user_input(prompt)

if __name__ == "__main__":
    main()
