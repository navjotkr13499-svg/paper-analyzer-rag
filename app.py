import sys
import os
import streamlit as st
import tempfile

# Fix module imports for Streamlit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_loader import PDFLoader
from src.rag_engine import RAGEngine

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

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    if st.sidebar.button("Process Documents"):
        pdf_loader = PDFLoader()
        
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                
                try:
                    # Process PDF: load → extract → chunk
                    chunks = pdf_loader.process_pdf(tmp.name)
                    
                    # Add to RAG engine
                    rag.add_documents(chunks, metadata={"source": file.name})
                    st.sidebar.success(f"✅ Processed: {file.name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing {file.name}: {str(e)}")
                finally:
                    os.unlink(tmp.name)
        
        st.sidebar.success("🎉 All documents processed!")

# Main Q&A interface
st.header("❓ Ask Questions")
question = st.text_input(
    "Enter your question:",
    placeholder="What is this paper about?"
)

if question:
    with st.spinner("🔍 Searching documents..."):
        try:
            result = rag.query(question)
            
            # Display answer
            st.subheader("✅ Answer")
            st.write(result.get("result", "No answer found"))
            
            # Display sources
            st.subheader("📄 Sources")
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"], 1):
                    with st.expander(f"Source {i}"):
                        st.write(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            else:
                st.info("No sources found")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.write("🔧 Built with LangChain + Chroma + OpenAI")
