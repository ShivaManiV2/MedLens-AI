import os
import json
import yaml
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.json")

class RetrievalPipeline:
    def __init__(self):
        # Initialize Vector Store
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection(name="medical_research")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load chunks for BM25 and initialize
        self.rebuild_bm25()
        
        # Initialize Cross-Encoder for Re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rebuild_bm25(self):
        with open(CHUNKS_PATH, "r") as f:
            self.chunks_data = json.load(f)
        tokenized_corpus = [doc["text"].lower().split() for doc in self.chunks_data]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def bm25_search(self, query, top_k=5):
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n_indices:
             if doc_scores[i] > 0:
                 results.append({
                     "id": self.chunks_data[i]["id"],
                     "text": self.chunks_data[i]["text"],
                     "metadata": self.chunks_data[i]["metadata"],
                     "score": doc_scores[i]
                 })
        return results

    def vector_search(self, query, top_k=5):
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        vec_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                vec_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]  # smaller is better for L2
                })
        return vec_results

    def retrieve_and_rerank(self, query):
        k_vec = config["retrieval"]["top_k_vector"]
        k_bm25 = config["retrieval"]["top_k_bm25"]
        k_final = config["retrieval"]["final_k_rerank"]
        
        vec_results = self.vector_search(query, top_k=k_vec)
        bm25_results = self.bm25_search(query, top_k=k_bm25)
        
        # Combine and deduplicate
        combined_dict = {}
        for r in vec_results + bm25_results:
            combined_dict[r["id"]] = r
            
        unique_chunks = list(combined_dict.values())
        
        if not unique_chunks:
            return []
            
        # Re-rank
        pairs = [[query, chunk["text"]] for chunk in unique_chunks]
        scores = self.cross_encoder.predict(pairs)
        
        for i, score in enumerate(scores):
            unique_chunks[i]["rerank_score"] = float(score)
            
        ranked_chunks = sorted(unique_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_chunks[:k_final]
