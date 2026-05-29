import os
import uuid
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

# 1. Initialize local Qdrant
client = QdrantClient(path="./qdrant_db")

def advanced_chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list:
    """
    A robust character chunker with overlap to prevent context loss.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if chunk_size <= overlap:
            break
            
    return chunks

class DocumentIngester:
    def __init__(self, collection_name="extraordinary_knowledge_base"):
        self.collection_name = collection_name
        self.client = client
        
        print("Initializing embedding model...")
        # FIX: Force threads=1 to prevent Windows multiprocessing crashes
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def ingest(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Chunks documents, generates embeddings, and cleanly upserts them into Qdrant.
        """
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            chunks = advanced_chunk_text(doc)
            all_chunks.extend(chunks)
            
            doc_meta = metadata[idx] if metadata is not None and idx < len(metadata) else {"source": f"doc_{idx}"}
            for i, chunk in enumerate(chunks):
                chunk_meta = doc_meta.copy()
                chunk_meta["chunk_index"] = str(i)
                all_metadata.append(chunk_meta)
                
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        
        # Ensure we evaluate the generator into a list in a single thread
        embeddings = list(self.embedding_model.embed(all_chunks))
        
        print("Constructing Qdrant points...")
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=embedding.tolist(),
                payload={"text": chunk, **all_metadata[i]} 
            )
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
        ]
        
        print("Upserting vectors into database...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Success! {len(points)} chunks safely stored in Qdrant.")

# --- Local Testing ---
if __name__ == "__main__":
    ingester = DocumentIngester()
    
    mock_docs = [
        "To configure firewall settings for the v2 server, navigate to the Securmin console, select Infirewall rules, add inbound rules for ports 80, 443, and 22, enable logging, and apply changes.",
        "Total revenue for Q3 2025 was approximately $4.2M, based on aggregated sales up to September 2025."
    ]
    
    metadata = [
        {"source": "server_runbook_v2.md", "type": "technical", "access_level": "admin"},
        {"source": "q3_financials.csv", "type": "business", "access_level": "executive"}
    ]
    
    ingester.ingest(mock_docs, metadata)