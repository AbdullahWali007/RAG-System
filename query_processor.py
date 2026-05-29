import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# 1. Define the Strict Data Structure for the Router
class QueryIntent(BaseModel):
    intent: str = Field(description="Must be one of: 'vector_search', 'sql_search', 'chitchat', or 'invalid'")
    hypothetical_answer: str = Field(description="A 2-3 sentence hypothetical perfect answer to the query. Leave empty if chitchat or invalid.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating how confident you are in this routing.")

class QueryProcessor:
    def __init__(self):
        # Initialize LongCat API using the OpenAI SDK format
        self.client = OpenAI(
            api_key=os.getenv("LONGCAT_API_KEY"),
            base_url="https://api.longcat.chat/openai/v1"
        )
        # FIX 1: Trying the lowercase model name, or the updated Exp model alias. 
        # If this still fails, swap this to "LongCat-Flash-Chat-2602-Exp"
        self.model = "LongCat-Flash-Chat-2602-Exp"

    # 2. Production Edge-Case Handling: Automatic Retries with Exponential Backoff
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def process_query(self, user_query: str) -> dict:
        """
        Takes a raw user query, routes it, and generates a HyDE (Hypothetical Document Embedding) string.
        """
        if not user_query or len(user_query.strip()) < 2:
            return {"intent": "invalid", "message": "Query too short or empty."}

        # FIX 2: Strengthened the prompt to guarantee pure JSON since we are removing the API flag
        system_prompt = (
            "You are the routing brain of an advanced RAG system. Your job is to analyze the user's query and output a strict JSON object.\n"
            "1. Determine the intent: 'vector_search' (for knowledge base questions), 'sql_search' (for exact data/metrics), or 'chitchat' (for greetings/general talk).\n"
            "2. If the intent is 'vector_search', write a 'hypothetical_answer'. This is a highly educated guess of what the answer might look like. We will use this to search our vector database.\n"
            "CRITICAL: Output ONLY valid JSON matching this schema: {\"intent\": \"string\", \"hypothetical_answer\": \"string\", \"confidence_score\": float}. "
            "Do not include markdown blocks, greetings, or any other text."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Query: {user_query}"}
                ],
                temperature=0.1, 
                max_tokens=500,
                # FIX 3: Removed response_format={"type": "json_object"} which often triggers 400 errors on proxy APIs
            )
            
            raw_output = (response.choices[0].message.content or "").strip()
            
            # FIX 4: Strip markdown formatting just in case the AI ignores the prompt instruction
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
                
            raw_output = raw_output.strip()

            parsed_json = json.loads(raw_output)
            validated_data = QueryIntent(**parsed_json)
            
            return validated_data.model_dump()

        except ValidationError as e:
            print(f"Validation Error: AI returned malformed structure. {e}")
            raise 
        except Exception as e:
            print(f"API Error: {e}")
            raise
# --- Local Testing ---
if __name__ == "__main__":
    processor = QueryProcessor()
    
    # Test 1: A standard knowledge-base question
    print("Testing Vector Search Intent...")
    print(processor.process_query("How do I configure the firewall settings for the new v2 server?"))
    print("-" * 40)
    
    # Test 2: A data/SQL question
    print("Testing SQL Search Intent...")
    print(processor.process_query("What was our total revenue for Q3 2025?"))
    print("-" * 40)
    
    # Test 3: Edge case - Gibberish
    print("Testing Gibberish/Invalid Intent...")
    print(processor.process_query("asdfasdfasdf"))