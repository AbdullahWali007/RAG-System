import os
from dotenv import load_dotenv
from openai import OpenAI

# Import the modules we built in Steps 1 and 3
from query_processor import QueryProcessor
from search_engine import RAGRetriever

# Load environment variables
load_dotenv()

class ExtraordinaryRAG:
    def __init__(self):
        print("Initializing the Extraordinary RAG Pipeline...")
        self.router = QueryProcessor()
        self.retriever = RAGRetriever()
        
        # Initialize LongCat for the final generation phase
        self.client = OpenAI(
            api_key=os.getenv("LONGCAT_API_KEY"),
            base_url="https://api.longcat.chat/openai/v1"
        )
        # FIX: Dynamically inherit the working model name from your router!
        # FIX: The proxy is rejecting the alias for standard generations. 
        # Forcing the explicit experimental release tag.
        self.model = "LongCat-Flash-Chat-2602-Exp"

    def ask(self, user_query: str) -> str:
        """
        The master pipeline: Route -> Retrieve -> Generate
        """
        print(f"\n[1/3] Analyzing intent for: '{user_query}'")
        intent_data = self.router.process_query(user_query)
        
        # EDGE CASE: Chit-chat or invalid queries
        if intent_data["intent"] in ["chitchat", "invalid"]:
            print("[System] Bypassing database (Chit-chat detected).")
            return self._handle_chitchat(user_query)
            
        if intent_data["intent"] == "sql_search":
            return "This requires structured data. SQL module not yet implemented!"

        # 2. Retrieve Context (Using the HyDE hallucinated answer for better vector matching)
        search_target = intent_data.get("hypothetical_answer") or user_query
        print("[2/3] Retrieving context from Qdrant...")
        matches = self.retriever.retrieve(search_target, limit=3)
        
        # EDGE CASE: Nothing found in the database
        if not matches or matches[0]["score"] < 0.3:
            return "I'm sorry, but I couldn't find any relevant information in our knowledge base."

        # 3. Format the retrieved context
        context_blocks = []
        for i, match in enumerate(matches):
            context_blocks.append(f"[Document {i+1} | Source: {match['source']}]\n{match['text']}")
        
        formatted_context = "\n\n".join(context_blocks)

        # 4. Final Generation with LongCat
        print("[3/3] Synthesizing final response via LongCat...")
        system_prompt = (
            "You are an elite enterprise AI assistant. Your prime directive is to answer the user's question "
            "based STRICTLY and EXCLUSIVELY on the provided context.\n"
            "- Do NOT use outside knowledge.\n"
            "- If the context does not contain the answer, say 'I do not have enough information.'\n"
            "- ALWAYS cite the Source file name in your response (e.g., 'According to server_runbook.md...')."
        )
        
        user_prompt = f"Context Information:\n{formatted_context}\n\nUser Query: {user_query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Keep it extremely low to prevent hallucinations
                max_tokens=1024
            )
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except Exception as e:
            return f"Generation Error: {e}"

    def _handle_chitchat(self, query: str) -> str:
        """Handles basic conversation without burning database compute."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep your answer under 2 sentences."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""

# --- Run the Application ---
if __name__ == "__main__":
    app = ExtraordinaryRAG()
    
    print("\n" + "="*50)
    print(" 🚀 EXTRAORDINARY RAG SYSTEM ONLINE")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nAsk a question (or type 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input.strip():
                continue
                
            answer = app.ask(user_input)
            
            print("\n" + "-"*50)
            print("🤖 AI RESPONSE:")
            print(answer)
            print("-"*50)
            
        except KeyboardInterrupt:
            break