"""
Streamlit Frontend for Paper Analyzer RAG
User-friendly interface for uploading papers and asking questions
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ============== Page Configuration ==============
st.set_page_config(
    page_title="📚 Paper Analyzer RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Styling ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3em;
        color: #555;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============== Header ==============
st.markdown('<div class="main-header">📚 Research Paper Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by RAG (Retrieval-Augmented Generation)</div>', unsafe_allow_html=True)

# ============== API Configuration ==============
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ============== Sidebar ==============
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="The FastAPI backend URL"
    )
    
    st.divider()
    
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Upload Papers**: Add PDF research papers
    2. **Ask Questions**: Query the papers naturally
    3. **Get Answers**: Receive AI-generated responses with sources
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Database", use_container_width=True):
        try:
            response = requests.delete(f"{api_url}/clear")
            if response.status_code == 200:
                st.success("✓ Database cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============== Main Content ==============
col1, col2 = st.columns(2)

# ============== Left Column: Upload ==============
with col1:
    st.subheader("📤 Upload Papers")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Select one or more PDF research papers"
    )
    
    if st.button("📌 Upload & Index", use_container_width=True):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        response = requests.post(
                            f"{api_url}/upload",
                            files=files
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.markdown(f"""
                            <div class="success-box">
                            ✓ {data['filename']}<br/>
                            📊 {data['chunks_processed']} chunks indexed
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Error: {response.json()['detail']}")
                    
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please select at least one PDF file")

# ============== Right Column: Query ==============
with col2:
    st.subheader("🔍 Ask Questions")
    
    question = st.text_area(
        "Your question about the papers:",
        placeholder="E.g., What are the main findings of the research?",
        height=100,
        help="Ask any question about the uploaded papers"
    )
    
    if st.button("📤 Get Answer", use_container_width=True):
        if question.strip():
            with st.spinner("Searching papers and generating answer..."):
                try:
                    response = requests.post(
                        f"{api_url}/query",
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.markdown(data["answer"])
                        
                        # Display sources
                        if data["sources"]:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(data["sources"], 1):
                                with st.expander(f"Source {i} - {source['metadata'].get('filename', 'Unknown')}"):
                                    st.markdown(source["content"])
                    else:
                        error_msg = response.json().get('detail', 'Unknown error')
                        st.markdown(f"""
                        <div class="error-box">
                        ❌ Error: {error_msg}
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                    ❌ Connection Error: {str(e)}<br/>
                    Make sure the FastAPI server is running!
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a question")

# ============== Footer ==============
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>Built with LangChain • OpenAI • Chroma • FastAPI • Streamlit</p>
    <p>🚀 AI-Powered Research Paper Analysis</p>
</div>
""", unsafe_allow_html=True)
