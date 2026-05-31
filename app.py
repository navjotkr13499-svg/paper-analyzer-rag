import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor
import tempfile
import os

st.set_page_config(page_title="Paper Analyzer RAG", layout="wide")

st.title("📚 Paper Analyzer - RAG System")
st.write("Upload research papers and ask questions powered by AI!")

# Initialize RAG engine
@st.cache_resource
def load_rag():
    return RAGEngine()

rag = load_rag()

# Sidebar for file upload
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=['pdf'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    if st.sidebar.button("Process Documents"):
        processor = DocumentProcessor()
        
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                
                # Process PDF
                documents = processor.load_pdf(tmp.name)
                chunks = processor.chunk_documents(documents)
                rag.add_documents(chunks, metadata={"source": file.name})
                
                st.sidebar.write(f"✅ Processed: {file.name}")
                os.unlink(tmp.name)
        
        st.sidebar.success("All documents processed!")

# Main Q&A interface
st.header("❓ Ask Questions")
question = st.text_input("Enter your question:", placeholder="What is RAG and how does it work?")

if question:
    with st.spinner("🔍 Searching documents..."):
        result = rag.query(question)
        
        # Display answer
        st.subheader("✅ Answer")
        st.write(result["result"])
        
        # Display sources
        st.subheader("📄 Sources")
        if result.get("source_documents"):
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"Source {i}"):
                    st.write(doc.page_content)
        else:
            st.info("No sources found")

st.markdown("---")
st.write("Built with LangChain + Chroma + OpenAI")
