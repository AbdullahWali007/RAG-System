**Extraordinary RAG** is an advanced, modular Retrieval-Augmented Generation system engineered for highly accurate, hallucination-free document intelligence. The architecture handles both unstructured documents (via character-based chunking with semantic overlap) and structured relational knowledge bases like CSV-formatted Q&A datasets.

#### **Core Architectural Highlights**

* **Dual-Engine LLM Support:** Fully integrated with the **LongCat API** for rapid cloud-based generation, with complete native support to swap in **Ollama** for 100% local, offline, and privacy-focused LLM inference.
* **Intelligent Intent Routing:** A pre-retrieval routing layer automatically classifies queries into conversational chit-chat, structured database tracking, or semantic knowledge-base searches to minimize API latency and compute costs.
* **HyDE (Hypothetical Document Embeddings):** The search layer utilizes a generation-guided retrieval technique, transforming raw user questions into rich hypothetical answers before database lookup to drastically improve vector similarity matching.
* **Single-Threaded Qdrant Integration:** Configured specifically for multi-core Windows environments using a local Qdrant instance and single-threaded `fastembed` processing to entirely bypass concurrent resource locks.
* **Strict Safety Rails:** Armed with specialized system-prompt configurations that force the LLM to cleanly refuse out-of-bounds questions rather than hallucinating facts, while guaranteeing precise row-level or file-level source citations for every answer.

---

### **How to Swap Between LongCat and Ollama**

You can easily configure the system to run in the cloud or completely on your local machine.

#### **Option A: Cloud Inference (LongCat)**

Ensure your `.env` file contains your API credentials:

```env
LONGCAT_API_KEY=your_api_key_here

```

#### **Option B: Local Inference (Ollama)**

1. Install and run [Ollama](https://ollama.com/).
2. Pull your model of choice (e.g., `ollama pull llama3` or `ollama pull misatral`).
3. Point your initialization client in `main.py` and `query_processor.py` to your local host:

```python
self.client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # Ollama does not require a real token
)
self.model = "llama3" # Or your preferred local model

```
