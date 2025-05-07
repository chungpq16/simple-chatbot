import os
import hashlib
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from bs4 import BeautifulSoup
import lxml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self, 
        docs_dir: str = "./docs", 
        db_dir: str = "./vectordb", 
        metadata_file: str = "./vectordb/metadata.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "nomic-embed-text:latest"
    ):
        self.docs_dir = os.path.abspath(docs_dir)
        self.db_dir = os.path.abspath(db_dir)
        self.metadata_file = os.path.abspath(metadata_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.file_metadata = self._load_metadata()
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
    def _load_metadata(self) -> Dict:
        """Load metadata about processed files or initialize if it doesn't exist"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata about processed files"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.file_metadata, f, indent=2)
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file to detect changes"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _get_file_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify new, modified, and removed files
        Returns: (new_files, modified_files, removed_files)
        """
        current_files = set()
        new_files = []
        modified_files = []
        
        # Check for new or modified files
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    filepath = os.path.join(root, file)
                    current_files.add(filepath)
                    
                    file_hash = self._calculate_file_hash(filepath)
                    rel_path = os.path.relpath(filepath, self.docs_dir)
                    
                    if rel_path not in self.file_metadata:
                        new_files.append(filepath)
                        self.file_metadata[rel_path] = {
                            'hash': file_hash,
                            'last_processed': datetime.now().isoformat()
                        }
                    elif self.file_metadata[rel_path]['hash'] != file_hash:
                        modified_files.append(filepath)
                        self.file_metadata[rel_path]['hash'] = file_hash
                        self.file_metadata[rel_path]['last_processed'] = datetime.now().isoformat()
        
        # Check for removed files
        stored_files = set()
        removed_files = []
        
        for rel_path in list(self.file_metadata.keys()):
            filepath = os.path.join(self.docs_dir, rel_path)
            stored_files.add(filepath)
            
            if filepath not in current_files:
                removed_files.append(filepath)
                del self.file_metadata[rel_path]
        
        return new_files, modified_files, removed_files
    
    def _process_pdf(self, filepath: str) -> List:
        """Process a PDF file, extract text and hyperlinks"""
        logger.info(f"Processing PDF: {filepath}")
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        
        # Add file metadata to each document
        for doc in documents:
            doc.metadata['source'] = filepath
            doc.metadata['filename'] = os.path.basename(filepath)
        
        # Extract hyperlinks (this is a simplified approach)
        # A full implementation would need to parse PDF structure more deeply
        # using libraries like PyMuPDF or pdfplumber
        for doc in documents:
            try:
                # Simple approach using BeautifulSoup to find anything that looks like a URL
                soup = BeautifulSoup(doc.page_content, 'lxml')
                links = []
                for text in soup.stripped_strings:
                    # Very naive hyperlink detection - in production use better URL detection
                    if "http://" in text or "https://" in text:
                        links.append(text)
                
                if links:
                    # Convert list of hyperlinks to a string to avoid metadata type errors
                    doc.metadata['hyperlinks'] = ', '.join(links)
            except Exception as e:
                logger.warning(f"Error extracting hyperlinks from {filepath}: {e}")
        
        return documents
    
    def _chunk_documents(self, documents: List) -> List:
        """Split documents into chunks for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def process_documents(self) -> Optional[Chroma]:
        """
        Process all documents in the docs directory
        Returns a Chroma vector store or None if no documents were processed
        """
        new_files, modified_files, removed_files = self._get_file_changes()
        
        if not new_files and not modified_files and not removed_files:
            logger.info("No document changes detected.")
            
            # If vectorstore exists, load and return it
            if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
                return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            return None
        
        logger.info(f"Found {len(new_files)} new files, {len(modified_files)} modified files, and {len(removed_files)} removed files.")
        
        # Process new and modified files
        all_chunks = []
        files_to_process = new_files + modified_files
        
        for filepath in files_to_process:
            documents = self._process_pdf(filepath)
            chunks = self._chunk_documents(documents)
            all_chunks.extend(chunks)
            logger.info(f"Processed {filepath}: {len(chunks)} chunks extracted")
        
        # If there are changes, rebuild the vectorstore
        if all_chunks:
            # Handle incremental updates more intelligently in a production system
            # Here we're rebuilding from scratch if any files changed, which is inefficient
            # A better approach would track document IDs and selectively update
            
            # If we have existing content AND only some documents changed, we should:
            # 1. Load existing vectorstore
            # 2. Remove vectors for changed/deleted files
            # 3. Add vectors for new/changed files
            # This approach would be more efficient but is more complex
            
            logger.info(f"Building vector store with {len(all_chunks)} chunks")
            vectorstore = Chroma.from_documents(
                documents=all_chunks, 
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            
            # Persist to disk
            vectorstore.persist()
            logger.info(f"Vector store built and persisted to {self.db_dir}")
            
            # Save metadata
            self._save_metadata()
            
            return vectorstore
        else:
            # If there were only removals
            return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
    
    def get_vectorstore(self) -> Chroma:
        """Get a reference to the vector store"""
        # Process documents if needed and return the vectorstore
        vectorstore = self.process_documents()
        if vectorstore:
            return vectorstore
        
        # If processing returns None (no changes and no existing store)
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        
        raise FileNotFoundError("No documents processed and no existing vector store found.")