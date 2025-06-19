import streamlit as st
import os
import tempfile
import time
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Page configuration
st.set_page_config(
    page_title="PDF QA Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())[:8]

def get_api_keys():
    """Get API keys from Streamlit secrets or user input"""
    openai_key = None
    pinecone_key = None
    
    # Try to get from secrets first
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        return openai_key, pinecone_key, True
    except (KeyError, FileNotFoundError):
        # Fall back to user input
        return None, None, False

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_pinecone(pinecone_key):
    """Setup Pinecone vector database with user session isolation"""
    try:
        os.environ["PINECONE_API_KEY"] = pinecone_key
        pc = Pinecone(api_key=pinecone_key)
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        
        # Create unique index name for this session
        index_name = f"qa-bot-{st.session_state.user_session_id}"
        existing_indexes = [item['name'] for item in pc.list_indexes().indexes]
        
        if index_name in existing_indexes:
            pc.delete_index(index_name)
            time.sleep(10)  # Wait for deletion
        
        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
        
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        index = pc.Index(index_name)
        embeddings = OpenAIEmbeddings()
        
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        return vectorstore, index_name
    
    except Exception as e:
        st.error(f"Error setting up Pinecone: {str(e)}")
        return None, None

def process_pdf(uploaded_file, vectorstore):
    """Process uploaded PDF file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process PDF
        pdf_loader = PyPDFLoader(tmp_file_path)
        pages = pdf_loader.load_and_split()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Add documents to vector store
        vectorstore.add_documents(documents=splits)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return True, len(splits)
    
    except Exception as e:
        return False, str(e)

def ask_question(question, vectorstore):
    """Get answer to question using RAG"""
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )
        
        prompt_rag = PromptTemplate.from_template(
            "Answer the question: {question} based on the following context: {context}. "
            "Provide references to the source material when possible. "
            "If the answer cannot be found in the context, say so clearly."
        )
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            max_tokens=1000
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        result = rag_chain.invoke(question)
        return result.content
    
    except Exception as e:
        return f"Error processing question: {str(e)}"

def cleanup_resources():
    """Cleanup Pinecone resources on session end"""
    try:
        if 'index_name' in st.session_state and st.session_state.get('pinecone_key'):
            pc = Pinecone(api_key=st.session_state.pinecone_key)
            if st.session_state.index_name in [item['name'] for item in pc.list_indexes().indexes]:
                pc.delete_index(st.session_state.index_name)
    except:
        pass  # Silently fail cleanup

# Main app
def main():
    st.title("📄 PDF Question Answering System")
    st.markdown("Upload a PDF document and ask questions about its content using AI.")
    
    # Get API keys
    openai_key, pinecone_key, using_secrets = get_api_keys()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        if not using_secrets:
            st.warning("API keys not found in secrets. Please enter them below:")
            
            openai_key = st.text_input(
                "🔑 OpenAI API Key", 
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            
            pinecone_key = st.text_input(
                "🌲 Pinecone API Key", 
                type="password",
                help="Get your API key from https://app.pinecone.io/"
            )
        else:
            st.success("✅ API keys loaded from secrets")
        
        # Session info
        st.markdown("---")
        st.markdown(f"**Session ID:** `{st.session_state.user_session_id}`")
        
        if st.button("🔄 Reset Session"):
            # Cleanup current session
            cleanup_resources()
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if openai_key and pinecone_key:
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Initialize vector store if not done
        if not st.session_state.vectorstore:
            with st.spinner("Initializing vector database..."):
                vectorstore, index_name = setup_pinecone(pinecone_key)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.index_name = index_name
                    st.session_state.pinecone_key = pinecone_key
                    st.success("✅ Vector database initialized!")
                else:
                    st.error("❌ Failed to initialize vector database")
                    return
        
        # File upload section
        st.subheader("📎 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            process_btn = st.button("🚀 Process Document", disabled=not uploaded_file)
        
        if process_btn and uploaded_file:
            with st.spinner("Processing PDF document..."):
                success, result = process_pdf(uploaded_file, st.session_state.vectorstore)
            
            if success:
                st.success(f"✅ Document processed successfully! Created {result} text chunks.")
                st.session_state.documents_processed = True
            else:
                st.error(f"❌ Error processing document: {result}")
        
        # Question answering section
        if st.session_state.documents_processed:
            st.subheader("❓ Ask Questions")
            
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is this document about?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_btn = st.button("🔍 Get Answer", disabled=not question)
            
            if ask_btn and question:
                with st.spinner("Searching for answer..."):
                    answer = ask_question(question, st.session_state.vectorstore)
                
                st.subheader("💬 Answer:")
                st.write(answer)
                
                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Chat history
        if st.session_state.get('chat_history'):
            with st.expander("📜 Chat History"):
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.markdown("---")
    
    else:
        st.warning("🔐 Please provide API keys to start using the application.")
        
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            ### Getting Started
            
            1. **Get API Keys:**
               - OpenAI: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
               - Pinecone: Visit [app.pinecone.io](https://app.pinecone.io/)
            
            2. **For Streamlit Cloud Deployment:**
               - Add keys to your app's secrets in the Streamlit Cloud dashboard
               - Format: `OPENAI_API_KEY = "your-key-here"`
            
            3. **Usage:**
               - Upload a PDF document
               - Click "Process Document"
               - Ask questions about the content
            """)

if __name__ == "__main__":
    main()