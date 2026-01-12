# Agentic RAG Backend – Assignment Submission

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) AI Agent** capable of answering user queries with accurate, source-attributed responses. The system integrates a **large language model**, internal document retrieval, and external web search tools, exposed via a **FastAPI backend API**.

---

## Task 1 – AI Agent Development

**Objective:**  
Build an AI agent that can decide whether to answer directly or retrieve information from documents, and return a structured response.

**Implementation Highlights:**

- **LLM:** `ChatGoogleGenerativeAI` (Gemini 2.5 Flash)
- **Prompt engineering:** Strict JSON schema enforced via `PydanticOutputParser`  
- **Tools:**  
  - `rag_search` – queries internal company documents using FAISS embeddings  
  - `search_web` – queries DuckDuckGo for external or real-time information
- **Memory:** Session-based memory using `InMemorySaver`  
- **Output:** JSON with fields `topic`, `summary`, `sources`, `tools_used`  

**Result:**  
Agent automatically decides when to fetch from internal docs or answer directly, always returning structured JSON.

---

## Task 2 – RAG (Retrieval-Augmented Generation)

**Objective:**  
Embed and store documents, retrieve relevant chunks, and provide them as context to the LLM.

**Implementation Highlights:**

- **Documents:** Sample internal text files (`rag/docs/*.txt`) covering company policies, FAQs, and technical documentation
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **Vector Store:** FAISS
- **Text splitting:** `RecursiveCharacterTextSplitter` (chunk_size=300, overlap=50)
- **Metadata:** Each document chunk includes `metadata["source"]` for accurate source attribution
- **Retrieval:** Top 3 most similar chunks returned to the agent

## Future Improvements

### Local LLM–Based Agentic RAG

A significant future enhancement for this system is the integration of a **fully local Large Language Model (LLM)** to run the complete **Agentic RAG pipeline** on a local server.

**Planned Enhancements:**

- Integrate local LLMs such as **LLaMA / LLaMA 3, Mistral, Phi, or Qwen**
- Serve models using frameworks like **Ollama**, **vLLM**, or **LM Studio**
- Run the entire RAG workflow locally:
  - Embedding generation
  - Vector similarity search using **FAISS**
  - Agent reasoning and response generation
- Enable **offline and private-network deployments**

**Benefits:**

- **Reduced operational cost** by eliminating cloud-based per-token API usage
- **Enhanced data privacy and security**, as internal documents remain within local infrastructure
- **Greater control and customization** over model behavior and performance
- Improved suitability for **enterprise, on-premise, and confidential use cases**

This improvement would make the system a **cost-efficient, privacy-preserving, enterprise-ready Agentic RAG solution** capable of running entirely on local infrastructure.



**Example Output:**
```json
{
  "answer": "Employees are entitled to 18 paid leave days per year...",
  "source": ["company_policy.txt"]
}

