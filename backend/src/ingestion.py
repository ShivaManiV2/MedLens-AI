import os
import glob
import yaml
import fitz  # PyMuPDF
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Use ChromaDB
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.json")

def extract_text_and_metadata_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
        
    meta = doc.metadata or {}
    
    # Try to extract the real title, otherwise fall back to the filename
    title = meta.get("title")
    if not title:
        title = os.path.basename(pdf_path)
        
    author = meta.get("author", "Unknown Author")
    
    # Sometimes PyMuPDF gets creationDate in a weird format, let's just grab the first 4 chars for Year
    creation_date = meta.get("creationDate", "")
    year = "Unknown Year"
    if creation_date.startswith("D:"):
        year = creation_date[2:6]
        
    metadata = {
        "title": title,
        "authors": author,
        "year": year,
        "journal": meta.get("subject", "General Medical Research"),
        "source": os.path.basename(pdf_path)
    }
    
    return text, metadata

if __name__ == "__main__":
        
    print("Starting ingestion...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", config["ingestion"]["data_dir"])
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"]
    )
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        text, metadata = extract_text_and_metadata_from_pdf(pdf_path)
        
        chunks = splitter.create_documents([text], metadatas=[metadata])
        all_chunks.extend(chunks)
        
    print(f"Extracted {len(all_chunks)} chunks.")
    
    # Store to ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name="medical_research")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    
    print("Computing embeddings and saving to ChromaDB...")
    # Compute embeddings inside add, but since we are using Gemini we will just compute them first
    embeds = embeddings.embed_documents(texts)
    collection.add(
        embeddings=embeds,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    # Save chunks for BM25 retrieval
    chunks_dict = []
    for i, chunk in enumerate(all_chunks):
        chunks_dict.append({
            "id": ids[i],
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks_dict, f)
        
    print("Ingestion complete!")
