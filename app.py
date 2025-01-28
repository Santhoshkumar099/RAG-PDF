import os
import streamlit as st
import fitz
from gtts import gTTS
import base64
import io
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def text_to_speech(text):
    """Convert text to speech using gTTS and return the audio bytes"""
    try:
        audio_buffer = io.BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def get_audio_html(audio_bytes):
    """Generate HTML for audio player"""
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
    return audio_html

def set_custom_style():
    st.markdown("""
        <style>
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stTextInput > div > div > input {
            padding-right: 40px;
        }
        .main-title {
            color: #1E88E5;
            text-align: center;
            padding-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .stButton > button {
            border-radius: 20px;
        }
        .answer-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            margin-bottom: 10px;
        }
        .audio-section {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            justify-content: center;
        }
        .audio-container {
            display: inline-block;
            margin-left: 10px;
        }
        div.row-widget.stButton {
            width: auto;
            min-width: 100px;
        }
        </style>
    """, unsafe_allow_html=True)

def get_pdf_processed(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    return text

def initialize_vector_store():
    pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs", key="process_button"):
        with st.spinner("Processing PDF files..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            st.session_state.docs = get_pdf_processed(pdf_files)
            st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
            st.session_state.vector = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
            st.success("PDFs processed successfully! You can now ask questions.")

def get_conversational_chain():
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only. 
        Please provide the most accurate response based on the question 
        <context> {context} </context> 
        Question: {input}"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector.as_retriever() if hasattr(st.session_state, 'vector') else None
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def user_input(prompt):
    if not hasattr(st.session_state, 'vector'):
        st.error("Please upload and process PDFs first!")
        return
    
    chain = get_conversational_chain()
    start = time.time()
    
    response = chain.invoke({"input": prompt})
    answer = response['answer']
    
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    st.session_state.current_answer = answer
    
    # Display answer with audio controls
    st.markdown("### Answer")
    
    # Create a container for the answer
    container = st.container()
    with container:
        # Display the answer
        st.markdown(f"""
            <div class="answer-container">
                {answer}
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for the audio section
        audio_cols = st.columns([0.3, 0.4, 0.3])
        
        # Use the middle column for audio controls
        with audio_cols[1]:
            # Create a container for horizontal layout
            st.markdown('<div class="audio-section">', unsafe_allow_html=True)
            
            # Add the Listen button
            if st.button("🔊 Listen", key="play_button"):
                audio_bytes = text_to_speech(answer)
                if audio_bytes:
                    # Display audio player inline
                    st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    end = time.time()
    st.write(f"Response time: {(end - start):.2f} seconds")
    
    with st.expander("View source in PDF"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    {doc.page_content}
                </div>
            """, unsafe_allow_html=True)

def main():
    set_custom_style()
    
    # Add PDF icon to the title
    st.markdown("""
        <h1 class="main-title">
            📄 Smart PDF Assistant
        </h1>
    """, unsafe_allow_html=True)
    
    initialize_vector_store()
    
    # Create input container with text input
    st.markdown("### Ask a Question:")
    prompt = st.text_input(
        "",
        placeholder="Type your question here",
        key="text_input_field"
    )
    
    # Process text input if available
    if prompt:
        user_input(prompt)

if __name__ == "__main__":
    main()
