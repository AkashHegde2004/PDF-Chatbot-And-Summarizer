import streamlit as st
import google.generativeai as genai
import os
import tempfile
from PyPDF2 import PdfReader
import re

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide"
)
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n\n"
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro-latest')


def get_gemini_answer(model, question, context):
    prompt = f"""
    Context information is below:
    {context}
    
    Given the context information and no prior knowledge, answer the question: {question}
    If the answer is not contained within the context, respond with "I don't have enough information to answer this question based on the provided PDF."
    """
    response = model.generate_content(prompt)
    return response.text


def summarize_with_gemini(model, text, word_count):
    prompt = f"""
    Summarize the following text in approximately {word_count} words:
    {text}
    
    The summary should be concise, accurate, and capture the main points of the text.
    """
    response = model.generate_content(prompt)
    return response.text

st.title("📄 PDF Question Answering & Summarizer")
st.markdown("Upload a PDF file, then ask questions about it or generate summaries of your desired length.")


api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")


uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file is not None:

    if st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.chat_history = []
        
        with st.spinner("Processing PDF..."):
        
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
        
            st.session_state.pdf_text = extract_text_from_pdf(tmp_path)
            
       
            os.unlink(tmp_path)
        
        st.sidebar.success(f"Processed: {uploaded_file.name}")
       
        word_count = len(st.session_state.pdf_text.split())
        st.sidebar.info(f"Word count: {word_count}")


tab1, tab2 = st.tabs(["Ask Questions", "Summarize PDF"])


with tab1:
    if st.session_state.pdf_text:
        st.header("Ask a Question")
        question = st.text_input("Enter your question about the PDF:")
        
        if st.button("Get Answer") and question and api_key:
            try:
                with st.spinner("Generating answer..."):
                    model = initialize_gemini(api_key)
                    answer = get_gemini_answer(model, question, st.session_state.pdf_text)
                    
                   
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                
                
                for item in st.session_state.chat_history:
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.divider()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "403" in str(e):
                    st.error("API key error. Please check your Gemini API key.")
    elif uploaded_file is None:
        st.info("Please upload a PDF file to get started.")
    else:
        st.info("PDF is being processed. Please wait.")


with tab2:
    if st.session_state.pdf_text:
        st.header("Generate Summary")
        summary_length = st.slider("Summary length (words)", min_value=50, max_value=500, value=100, step=50)
        
        if st.button("Generate Summary") and api_key:
            try:
                with st.spinner("Generating summary..."):
                    model = initialize_gemini(api_key)
                    summary = summarize_with_gemini(model, st.session_state.pdf_text, summary_length)
                
                st.subheader("Summary")
                st.markdown(summary)
                
               
                summary_word_count = len(summary.split())
                st.caption(f"Summary word count: {summary_word_count}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "403" in str(e):
                    st.error("API key error. Please check your Gemini API key.")
    elif uploaded_file is None:
        st.info("Please upload a PDF file to get started.")
    else:
        st.info("PDF is being processed. Please wait.")


with st.sidebar.expander("Instructions"):
    st.markdown("""
    1. Enter your Gemini API key
    2. Upload a PDF file
    3. Navigate to the tab for your desired function:
       - Ask questions about the PDF content
       - Generate summaries of specified length
    """)


with st.sidebar.expander("About"):
    st.markdown("""
    This app uses Google's Gemini API to process and analyze PDF documents.
    
    Features:
    - Extract text from PDF files
    - Answer questions based on PDF content
    - Generate customizable summaries
    """)


st.markdown("---")
st.caption("Powered by Gemini API and Streamlit")