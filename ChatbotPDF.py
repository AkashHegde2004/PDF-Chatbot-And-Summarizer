import streamlit as st
import PyPDF2
import io
from groq import Groq
import time
from typing import List, Dict
import re

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot & Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
    }
    
    .bot-message {
        background: #f1f3f4;
        color: #333;
        border-left: 4px solid #667eea;
    }
    
    .summary-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None

class PDFChatbot:
    def __init__(self, groq_client):
        self.client = groq_client
        self.model = "llama3-8b-8192"  # You can change this to other Groq models
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 4000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def summarize_text(self, text: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using Groq API"""
        if not text.strip():
            return "No text to summarize."
        
        summary_prompts = {
            "brief": "Provide a brief summary (2-3 sentences) of the following text:",
            "comprehensive": "Provide a comprehensive summary of the following text, highlighting key points and main ideas:",
            "bullet_points": "Summarize the following text in bullet points, focusing on the most important information:"
        }
        
        try:
            chunks = self.chunk_text(text)
            summaries = []
            
            for chunk in chunks:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries."},
                        {"role": "user", "content": f"{summary_prompts[summary_type]}\n\n{chunk}"}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                summaries.append(response.choices[0].message.content)
            
            # If multiple chunks, summarize the summaries
            if len(summaries) > 1:
                combined_summary = "\n\n".join(summaries)
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates coherent summaries from multiple parts."},
                        {"role": "user", "content": f"Combine and synthesize these summaries into one coherent summary:\n\n{combined_summary}"}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                return final_response.choices[0].message.content
            else:
                return summaries[0]
                
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def chat_with_pdf(self, question: str, pdf_text: str, chat_history: List[Dict]) -> str:
        """Chat with PDF content using Groq API"""
        if not pdf_text.strip():
            return "Please upload and process a PDF first."
        
        try:
            # Prepare context with chat history
            context_messages = [
                {"role": "system", "content": f"""You are a helpful AI assistant that answers questions based on the provided PDF content. 
                Use the following PDF content to answer questions accurately and helpfully.
                
                PDF Content:
                {pdf_text[:8000]}...  # Limit context size
                
                Guidelines:
                - Answer based on the PDF content provided
                - If information is not in the PDF, clearly state that
                - Be concise but comprehensive
                - Cite specific parts of the document when relevant"""}
            ]
            
            # Add recent chat history (last 5 exchanges)
            for msg in chat_history[-10:]:
                context_messages.append({"role": msg["role"], "content": msg["content"]})
            
            context_messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context_messages,
                max_tokens=1500,
                temperature=0.4
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 PDF Chatbot & Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Groq AI - Upload, Summarize, and Chat with your PDFs")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Get your API key from https://console.groq.com/"
        )
        
        if groq_api_key:
            try:
                st.session_state.groq_client = Groq(api_key=groq_api_key)
                st.success("✅ API Key configured!")
            except Exception as e:
                st.error(f"❌ Invalid API Key: {str(e)}")
        
        st.markdown("---")
        
        # Model selection
        model_options = {
            "Llama 3 8B": "llama3-8b-8192",
            "Llama 3 70B": "llama3-70b-8192",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Gemma 7B": "gemma-7b-it"
        }
        
        selected_model = st.selectbox(
            "Choose Model:",
            options=list(model_options.keys()),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Usage Stats")
        if 'chat_history' in st.session_state:
            st.metric("Messages", len(st.session_state.chat_history))
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to get started.")
        st.info("""
        To use this application:
        1. Get your free API key from [Groq Console](https://console.groq.com/)
        2. Enter the API key in the sidebar
        3. Upload a PDF file
        4. Start chatting with your document!
        """)
        return
    
    # Initialize chatbot
    chatbot = PDFChatbot(st.session_state.groq_client)
    chatbot.model = model_options[selected_model]
    
    # File upload section
    st.header("📎 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze and chat with"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing PDF..."):
            # Extract text from PDF
            pdf_text = chatbot.extract_text_from_pdf(uploaded_file)
            
            if pdf_text:
                st.session_state.pdf_text = pdf_text
                
                # Display PDF info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 File", uploaded_file.name)
                with col2:
                    st.metric("📝 Characters", f"{len(pdf_text):,}")
                with col3:
                    st.metric("📖 Words", f"{len(pdf_text.split()):,}")
                
                st.success("✅ PDF processed successfully!")
                
                # Summary section
                st.header("📋 Document Summary")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    summary_type = st.selectbox(
                        "Summary Type:",
                        ["comprehensive", "brief", "bullet_points"],
                        format_func=lambda x: x.replace("_", " ").title()
                    )
                
                with col2:
                    if st.button("📝 Generate Summary", use_container_width=True):
                        with st.spinner("🤖 Generating summary..."):
                            summary = chatbot.summarize_text(pdf_text, summary_type)
                            st.session_state.summary = summary
                
                # Display summary
                if hasattr(st.session_state, 'summary'):
                    st.markdown(f"""
                    <div class="summary-box">
                        <h3>📄 Document Summary</h3>
                        {st.session_state.summary}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat section
    if st.session_state.pdf_text:
        st.header("💬 Chat with PDF")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>AI:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a question about the PDF:",
                    placeholder="What is this document about?",
                    label_visibility="collapsed"
                )
            with col2:
                submit_button = st.form_submit_button("Send 🚀", use_container_width=True)
        
        if submit_button and user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Generate AI response
            with st.spinner("🤖 Thinking..."):
                ai_response = chatbot.chat_with_pdf(
                    user_question, 
                    st.session_state.pdf_text, 
                    st.session_state.chat_history
                )
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
            
            st.rerun()
        
        # Quick question suggestions
        if not st.session_state.chat_history:
            st.markdown("### 💡 Suggested Questions")
            suggestions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the main conclusions?",
                "Are there any important dates or numbers mentioned?"
            ]
            
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        # Add suggestion to chat
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": suggestion
                        })
                        
                        with st.spinner("🤖 Thinking..."):
                            ai_response = chatbot.chat_with_pdf(
                                suggestion, 
                                st.session_state.pdf_text, 
                                st.session_state.chat_history
                            )
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": ai_response
                            })
                        
                        st.rerun()
    
    else:
        st.info("📁 Upload a PDF file to start chatting with your document!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with ❤️ using Streamlit and Groq AI</p>
        <p>Need help? Check the <a href="https://console.groq.com/" target="_blank">Groq Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()