from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# 1. Connect to the existing local database
client = QdrantClient(path="./qdrant_db")

class RAGRetriever:
    def __init__(self, collection_name="kis_qa_knowledge_base"):
        self.collection_name = collection_name
        self.client = client
        
        # 2. Re-initialize the same model used during ingestion (Windows safe mode)
        print("Loading embedding model for retrieval...")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)

    def retrieve(self, query_text: str, limit: int = 2) -> list:
        """
        Converts query to vector and extracts the top matching raw text chunks using the modern API.
        """
        # 3. Embed the search query
        print(f"Embedding query: '{query_text}'")
        query_vector = list(self.embedding_model.embed([query_text]))[0].tolist()
        
        # 4. Perform semantic search in Qdrant using the new modern API
        print("Searching vector database...")
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True # Explicitly request the payload back
        ).points
        
        # 5. Format results cleanly
        retrieved_contexts = []
        for result in search_results:
            payload = result.payload or {}
            retrieved_contexts.append({
                "text": payload.get("text"),
                "source": payload.get("source"),
                "score": result.score
            })
            
        return retrieved_contexts

# --- Local Verification ---
if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Test query targeting our ingested data
    test_query = "How do I set up ports for the v2 server firewall?"
    
    print("\n" + "="*40)
    print(f"RUNNING RETRIEVAL TEST")
    print("="*40)
    
    matches = retriever.retrieve(test_query, limit=2)
    
    for i, match in enumerate(matches):
        print(f"\n[Match #{i+1}] (Confidence Score: {match['score']:.4f})")
        print(f"Source: {match['source']}")
        print(f"Content: {match['text']}")