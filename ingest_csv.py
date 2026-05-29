import csv
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

class CSV_Ingester:
    def __init__(self, collection_name="kis_qa_knowledge_base"):
        self.collection_name = collection_name
        self.client = QdrantClient(path="./qdrant_db")
        
        print("Loading embedding model (Threads=1 for Windows stability)...")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        
        # We create a fresh collection for this specific CSV data
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def ingest_csv(self, file_path: str, question_col: str = "Question", answer_col: str = "Answer"):
        """
        Reads a CSV, formats each row as a single context block, and upserts to Qdrant.
        """
        if not os.path.exists(file_path):
            print(f"Error: Could not find {file_path}. Make sure it is in the same folder!")
            return

        formatted_docs = []
        metadata = []
        
        print(f"Reading {file_path}...")
        
        # 1. Read the CSV and process row by row
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            
            # Auto-detect column names if they are slightly different (e.g., 'question', 'q', 'answer', 'a')
            headers = [h.lower().strip() for h in (reader.fieldnames or [])]
            fieldnames = reader.fieldnames or []
            actual_q_col = next((h for h in fieldnames if question_col.lower() in h.lower()), fieldnames[0] if fieldnames else question_col)
            actual_a_col = next((h for h in fieldnames if answer_col.lower() in h.lower()), fieldnames[1] if len(fieldnames) > 1 else (fieldnames[0] if fieldnames else answer_col))
            
            for row_idx, row in enumerate(reader):
                q_text = row.get(actual_q_col, "").strip()
                a_text = row.get(actual_a_col, "").strip()
                
                if not q_text or not a_text:
                    continue # Skip empty rows
                
                # Combine Q & A into a highly searchable text block
                combined_text = f"Question: {q_text}\nAnswer: {a_text}"
                formatted_docs.append(combined_text)
                
                # Store the exact row number so the LLM can cite it!
                metadata.append({"source": f"{file_path} (Row {row_idx + 2})"})

        print(f"Parsed {len(formatted_docs)} Q&A pairs. Generating embeddings...")
        
        # 2. Embed the combined strings
        embeddings = list(self.embedding_model.embed(formatted_docs))
        
        print("Constructing Qdrant points...")
        # 3. Create the database points
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=embedding.tolist(),
                payload={"text": doc, **metadata[i]} 
            )
            for i, (doc, embedding) in enumerate(zip(formatted_docs, embeddings))
        ]
        
        print("Upserting to Qdrant...")
        # Batch upsert in chunks of 100 to prevent memory spikes if the CSV is huge
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  -> Uploaded batch {i//batch_size + 1}...")
            
        print(f"Success! {len(points)} Q&A pairs from the CSV are safely stored in Qdrant.")

if __name__ == "__main__":
    ingester = CSV_Ingester()
    
    # Run the ingestion! (Adjust column names if your CSV header is different)
    ingester.ingest_csv(
        file_path="rag_sample_qas_from_kis.csv", 
        question_col="Question", # Update this if your header is "q" or "questions"
        answer_col="Answer"      # Update this if your header is "a" or "answers"
    )