import os
import sys
import logging
import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.document_processor import DocumentProcessor
from src.utils.agents import AgentGraph

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Agentic Document Chatbot",
    page_icon="📚",
    layout="wide"
)

def initialize_app():
    """Initialize the application components"""
    # Define paths
    docs_dir = os.path.join(project_root, "docs")
    db_dir = os.path.join(project_root, "vectordb")

    # Create directories if they don't exist
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    # Initialize document processor
    doc_processor = DocumentProcessor(
        docs_dir=docs_dir,
        db_dir=db_dir,
        embedding_model="nomic-embed-text:latest"
    )

    # Process documents (this will update only if needed)
    try:
        vectorstore = doc_processor.get_vectorstore()
        st.session_state.vectorstore = vectorstore
        
        # Check if any docs exist
        pdf_files = []
        for root, _, files in os.walk(docs_dir):
            pdf_files.extend([f for f in files if f.lower().endswith('.pdf')])
        
        # Store document names in session state
        st.session_state.pdf_files = pdf_files
        
        # Initialize agent graph with Azure OpenAI
        agent_graph = AgentGraph(
            vectorstore=vectorstore,
            model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),  # Use deployment name as model name
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            relevance_threshold=float(os.getenv("RELEVANCE_THRESHOLD", "0.7")),
            use_azure=True  # Flag to ensure only Azure OpenAI is used
        )
        
        st.session_state.agent_graph = agent_graph
        
        return True
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        st.error(f"Error: {str(e)}")
        return False

def show_sidebar():
    """Display the sidebar with document information and options"""
    st.sidebar.title("📚 Document Management")
    
    # Document status
    st.sidebar.header("Indexed Documents")
    if hasattr(st.session_state, "pdf_files") and st.session_state.pdf_files:
        for pdf in st.session_state.pdf_files:
            st.sidebar.text(f"- {pdf}")
    else:
        st.sidebar.warning("No PDFs found in the docs directory")
        
    st.sidebar.divider()
    
    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.info(
        """
        **How to use this chatbot:**
        
        1. Place your PDF documents in the 'docs' folder
        2. They will be automatically indexed
        3. Ask questions about the content of your documents
        4. The AI will collaborate with intelligent agents to find the best answer
        """
    )
    
    st.sidebar.divider()
    
    # Technical info
    st.sidebar.header("Technical Details")
    st.sidebar.markdown(
        """
        - Using Langchain for document processing
        - LangGraph for agentic workflow
        - Ollama for embeddings (nomic-embed-text)
        - Azure OpenAI for chat completion
        - Documents stored in Chroma vector database
        """
    )
    
    # Force reindex button
    if st.sidebar.button("Force Reindex Documents"):
        with st.spinner("Reindexing documents..."):
            try:
                # Load document processor and force reprocessing
                docs_dir = os.path.join(project_root, "docs")
                db_dir = os.path.join(project_root, "vectordb")
                metadata_file = os.path.join(db_dir, "metadata.json")
                
                # Force by deleting metadata
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    
                doc_processor = DocumentProcessor(
                    docs_dir=docs_dir,
                    db_dir=db_dir
                )
                vectorstore = doc_processor.get_vectorstore()
                st.session_state.vectorstore = vectorstore
                
                # Recreate agent graph with new vectorstore
                agent_graph = AgentGraph(vectorstore=vectorstore)
                st.session_state.agent_graph = agent_graph
                
                # Update file list
                pdf_files = []
                for root, _, files in os.walk(docs_dir):
                    pdf_files.extend([f for f in files if f.lower().endswith('.pdf')])
                st.session_state.pdf_files = pdf_files
                
                st.sidebar.success("Documents reindexed successfully!")
            except Exception as e:
                st.sidebar.error(f"Reindexing failed: {str(e)}")

def show_main_interface():
    """Display the main chatbot interface"""
    st.title("📑 Agentic Document Assistant")
    st.markdown(
        """
        I can answer questions about your PDF documents using an agentic approach with LangGraph.
        This means multiple AI agents collaborate to improve your results.
        """
    )
    
    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for user query
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            
            try:
                # Process with agent workflow
                if hasattr(st.session_state, "agent_graph"):
                    agent_graph = st.session_state.agent_graph
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Run agent workflow
                    result = agent_graph.run(prompt)
                    
                    # Extract results
                    answer = result.get("answer", "I couldn't find an answer in the documents.")
                    grade = result.get("quality_grade", 0.0)
                    final_query = result.get("final_query", prompt)
                    iterations = result.get("iterations", 0)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Display answer
                    message_placeholder.markdown(answer)
                    
                    # Add additional details in an expandable section
                    with st.expander("View processing details"):
                        st.markdown(f"**Quality Score:** {grade:.2f}/1.0")
                        st.markdown(f"**Final Query:** {final_query}")
                        st.markdown(f"**Iterations:** {iterations}")
                        st.markdown(f"**Processing Time:** {elapsed_time:.2f} seconds")
                else:
                    message_placeholder.markdown("Error: Agent graph not initialized properly.")
            except Exception as e:
                message_placeholder.markdown(f"Error generating response: {str(e)}")
                logger.error(f"Error: {str(e)}")
                return
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": message_placeholder.markdown})

def main():
    """Main application entrypoint"""
    # Check if the app is already initialized
    if "is_initialized" not in st.session_state:
        with st.spinner("Initializing app and processing documents..."):
            is_success = initialize_app()
            if is_success:
                st.session_state.is_initialized = True
            else:
                st.error("Failed to initialize the app. Please check logs.")
                return
    
    # Display the interface components
    show_sidebar()
    show_main_interface()

if __name__ == "__main__":
    main()