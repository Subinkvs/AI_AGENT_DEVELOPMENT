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

**Example Output:**
```json
{
  "answer": "Employees are entitled to 18 paid leave days per year...",
  "source": ["company_policy.txt"]
}
