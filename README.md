# Self-correcting Agentic RAG

This project presents a Retrieval-Augmented Generation system that intelligently combines multi-step document retrieval, dynamic query rewriting, document relevance grading, and web search augmentation. Designed as a self-correcting agent, it iteratively refines queries and supplements retrieved knowledge with real-time web data when existing documents fall short—delivering precise, contextually relevant answers to user questions.

---

## Features

- **Document Loading and Preprocessing:** Load PDFs and split them into chunks for retrieval.
- **Vector Store Initialization:** Embed documents using OpenAI embeddings and index them for similarity search.
- **Ensemble Retriever:** Combines parent document retriever and keyword-based retriever for better results.
- **Document Grading:** Scores retrieved documents by relevance to the question and filters out low-quality ones.
- **Query Rewriting:** Improves user queries for better retrieval when documents are insufficient.
- **Web Search Integration:** Adds recent and relevant web search results when the retrieved documents are not relevant.
- **RAG Chain for Generation:** Generates answers from retrieved documents and the rewritten query.
- **Flexible StateGraph Workflow:** Controls the multi-step process from retrieval to final generation with conditional branching.

---

## Setup Instructions

### Requirements

- Python 3.8+
- OpenAI API Key
- Required Python packages:
  - `openai`
  - `langchain`
  - `some_vectorstore_library` (your vector store choice, e.g., Chromadb, FAISS)
  - `tavily` or your web search API client

### Installation:

1. Clone the Repository:
```bash
git clone https://github.com/razamehar/Self-Correcting-Agentic-RAG-Corrective-RAG.git
cd Self-Correcting-Agentic-RAG-Corrective-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Contact 
For any questions or clarifications, please contact Raza Mehar at [raza.mehar@gmail.com].