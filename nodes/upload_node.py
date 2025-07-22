import os
import hashlib
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
import chromadb # Import the chromadb client to manage collections

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": "custom",
    ".xlsx": "custom",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
CHROMA_DB_PERSIST_DIR = "./chroma_db_files" # Base directory for all file-specific collections

# Helper to calculate file hash
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(4096), b""):  #This reads a file in binary mode ('rb') and in chunks (4096 bytes at a time).
            hasher.update(chunk) #The chunks are fed into an MD5 hasher to create a fingerprint of the file contents.
    return hasher.hexdigest() # hex string that uniquely identifies the file content

# Helper to generate a valid, unique collection name
def generate_collection_name(file_path: str, file_hash: str) -> str:
    return f"doc_{file_hash}"


def upload(state: Dict) -> Dict:
    file_path = state.get("file_path", "")
    suffix = os.path.splitext(file_path)[1].lower()

    if not file_path:
        logging.error("No file_path provided in state.")
        return {**state, "is_image": False, "documents": [], "vectordb": None, "answer": "Error: No file selected for upload.", "last_processed_file_path": None, "last_processed_file_hash": None}

    current_file_content_hash = calculate_file_hash(file_path)
    current_collection_name = generate_collection_name(file_path, current_file_content_hash)

    # --- Optimization Check: Is this file already processed and loaded into state? ---
    # This check is for subsequent queries on the SAME file within the SAME session
    if state.get("last_processed_file_hash") == current_file_content_hash and state.get("vectordb") is not None:
        logging.info(f"Content hash matches for '{current_collection_name}', and VectorDB is already in state. Skipping re-processing.")
        return {**state} # Return current state if no re-processing needed

    # --- Initialize ChromaDB client ---
    # We use PersistentClient to access collections from the persist_directory
    client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_DIR)
    
    # Check if the specific collection for this file already exists on disk
    # This is crucial for handling re-uploads of DIFFERENT documents
    collection_exists_on_disk = False
    try:
        # get_collection will raise an exception if it doesn't exist
        # We don't need the collection object here, just to know if it exists
        client.get_collection(name=current_collection_name)
        collection_exists_on_disk = True
        logging.info(f"Collection '{current_collection_name}' already exists on disk.")
    except Exception as e:
        logging.info(f"Collection '{current_collection_name}' does not exist on disk or error accessing: {e}")
        collection_exists_on_disk = False # Ensure it's false on any error

    vectordb = None

    if collection_exists_on_disk:
        # If collection exists, just load it for use
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(
            client=client, # Pass the client to ensure it uses the persistent client
            collection_name=current_collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PERSIST_DIR # Important: persist_directory must match the client's path
        )
        logging.info(f"Loaded existing ChromaDB collection '{current_collection_name}'.")
    else:
        # If it doesn't exist, we need to process the file and create the collection
        
        # If it's an image
        if suffix in IMAGE_EXTENSIONS:
            logging.info(f"Detected image file: {file_path}. No vector DB created for images.")
            return {**state, "is_image": True, "documents": [], "vectordb": None, "last_processed_file_path": file_path, "last_processed_file_hash": current_file_content_hash}

        # Else it's a document
        loader_type = LOADER_MAP.get(suffix)

        if not loader_type:
            logging.error(f"Unsupported file type: {suffix}")
            return {
                **state,
                "answer": f"Error: Unsupported file type: {suffix}"
            }

        docs = []
        if loader_type == "custom" and suffix in [".csv", ".xlsx"]:
            logging.info(f"Processing tabular file: {file_path}")
            try:
                import pandas as pd
                if suffix == ".csv":
                    df = pd.read_csv(file_path)
                elif suffix == ".xlsx" or suffix == ".xls":
                    df = pd.read_excel(file_path)

                row_strings = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
                content = "\n".join(row_strings)

                from langchain.schema import Document
                docs = [Document(page_content=content)]
            except Exception as e:
                logging.error(f"Tabular file load error: {e}")
                return {
                    **state,
                    "is_image": False,
                    "documents": [],
                    "vectordb": None,
                    "answer": f"Error loading {suffix.upper()} file: {e}",
                    "last_processed_file_path": file_path,
                    "last_processed_file_hash": None,
                }
        elif loader_type == PyPDFLoader and suffix in ['.pdf']:
            logging.info(f"Loading document file: {file_path} using {loader_type.__name__}")
            try:
                loader = loader_type(file_path)
                docs = loader.load()
            except Exception as e:
                logging.error(f"Document load error: {e}")
                return {
                    **state,
                    "is_image": False,
                    "documents": [],
                    "vectordb": None,
                    "answer": f"Error loading document: {e}",
                    "last_processed_file_path": file_path,
                    "last_processed_file_hash": None,
                }

        elif loader_type == TextLoader and suffix in ['.txt']:
            logging.info(f"Loading document file: {file_path} using {loader_type.__name__}")
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                

            except Exception as e:
                logging.error(f"Document load error: {e}")
                return {
                    **state,
                    "is_image": False,
                    "documents": [],
                    "vectordb": None,
                    "answer": f"Error loading document: {e}",
                    "last_processed_file_path": file_path,
                    "last_processed_file_hash": None,
                }

           


        if not docs:
            logging.error("Document loading failed or document is empty.")
            return {**state, "is_image": False, "documents": [], "vectordb": None, "answer": "Error: Document loading failed or document is empty.", "last_processed_file_path": file_path, "last_processed_file_hash": current_file_content_hash}

        logging.info("Splitting and embedding documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create the collection and add documents
        # Note: Chroma.from_documents internally calls client.get_or_create_collection
        # so this is the correct way to create and populate it if it doesn't exist.
        vectordb = Chroma.from_documents(
            documents=chunks,
            collection_name=current_collection_name,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PERSIST_DIR # This needs to match the client path
        )
        logging.info(f"New collection '{current_collection_name}' created and documents processed.")

    return {
        **state,
        "is_image": False,
        "documents": chunks if 'chunks' in locals() else [], # Only pass chunks if they were just generated
        "vectordb": vectordb,
        "last_processed_file_path": file_path,
        "last_processed_file_hash": current_file_content_hash,
        "active_collection_name": current_collection_name # Store the name of the active collection
    }