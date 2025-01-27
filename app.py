import os
import streamlit as st
import fitz
import pyttsx3
import speech_recognition as sr
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
from threading import Thread

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class TTSEngine:
    def __init__(self):
        self.engine = None
        self.is_speaking = False

    def init_engine(self):
        if not self.engine:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        return self.engine

    def speak(self, text):
        self.is_speaking = True
        engine = self.init_engine()
        engine.say(text)
        engine.runAndWait()
        self.is_speaking = False

    def stop(self):
        if self.engine and self.is_speaking:
            self.engine.stop()
            self.is_speaking = False
            self.engine = None

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
    
    if st.button("Submit & Process"):
        with st.spinner("Loading PDF..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            st.session_state.docs = get_pdf_processed(pdf_files)
            st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
            st.session_state.vector = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
            st.success("PDF content loaded successfully!")

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

def speech_to_text():
    """Capture speech input from microphone"""
    recognizer = sr.Recognizer()
    
    status_placeholder = st.empty()
    status_placeholder.write("Listening... Speak your question")
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            
            status_placeholder.write("Processing speech...")
            prompt = recognizer.recognize_google(audio)
            
            status_placeholder.write(f"Recognized: {prompt}")
            return prompt
            
    except sr.UnknownValueError:
        status_placeholder.error("Sorry, could not understand audio")
        return None
    except sr.RequestError as e:
        status_placeholder.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        status_placeholder.error(f"An error occurred: {e}")
        return None

def user_input(prompt):
    if not hasattr(st.session_state, 'vector'):
        st.error("Please upload and process PDFs first!")
        return
    
    # Initialize TTS engine in session state if not exists
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = TTSEngine()
    
    chain = get_conversational_chain()
    start = time.process_time()
    
    response = chain.invoke({"input": prompt})
    answer = response['answer']
    
    # Store the answer in session state
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    st.session_state.current_answer = answer
    
    # Create columns for the answer and buttons
    col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
    
    with col1:
        st.write(answer)
    
    with col2:
        if st.button("🔊 Play", key="play_button"):
            # Stop any existing playback
            st.session_state.tts_engine.stop()
            # Start new playback in a thread
            thread = Thread(target=st.session_state.tts_engine.speak, args=(answer,))
            thread.start()
    
    with col3:
        if st.button("⏹ Stop", key="stop_button"):
            st.session_state.tts_engine.stop()
    
    st.write("Response time: ", time.process_time() - start)
    
    with st.expander("Expand to view your question in PDF"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------")

def main():
    st.title("PDF Q&A with Voice Interaction")
    
    # Initialize session state
    if 'speech_input' not in st.session_state:
        st.session_state.speech_input = None
    
    initialize_vector_store()
    
    input_method = st.radio("Choose input method:", 
                           ["Text Input", "Speech Input"])
    
    if input_method == "Text Input":
        prompt = st.text_input("Input your question here")
        if prompt:
            user_input(prompt)
    else:
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            if st.button("Start Listening", key="listen_button"):
                st.session_state.speech_input = speech_to_text()
        
        if st.session_state.speech_input:
            st.write("Processing question:", st.session_state.speech_input)
            user_input(st.session_state.speech_input)

if __name__ == "__main__":
    main()
