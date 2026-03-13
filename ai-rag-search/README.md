# Endee AI RAG Semantic Search Demo

## Project Title
Endee AI RAG Semantic Search with FastAPI + Streamlit

## Problem Statement
Build a real-world semantic search + Retrieval Augmented Generation (RAG) system using Endee as the vector database to store embeddings and retrieve context for question answering.

## System Architecture
User → UI (Streamlit) → FastAPI → Embedding Model (sentence-transformers) → Endee Vector DB → Retrieved Context → LLM (OpenAI GPT) → Answer

## How Endee Vector Database is used
- documents are embedded using `all-MiniLM-L6-v2`
- vectors are inserted into Endee index `docs_index`
- semantic search queries are converted to vectors and run through `/api/v1/index/docs_index/search`
- retrieved metadata is used as RAG context for generation

## Installation Steps
1. Clone this repository and navigate to root:
   ```bash
   git clone https://github.com/endee-io/endee.git
   cd endee
   ```
2. Create and activate a Python environment (recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate         # windows
   source .venv/bin/activate         # unix
   ```
3. Install dependencies:
   ```bash
   pip install -r ai-rag-search/requirements.txt
   ```

## How to Run Endee
From repository root:
```bash
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
```
This starts Endee server on `http://localhost:8080`.

## How to Run the AI App
### Start FastAPI backend
```bash
cd ai-rag-search
python app.py
```
It starts on `http://localhost:8000`.

### Upload documents
```bash
curl -X POST http://localhost:8000/upload-documents
```

### Ask a question via API
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is RAG?\"}"
```

### Start Streamlit UI
```bash
cd ai-rag-search
streamlit run app.py
```

## Example Queries
- "What is a vector database and why is it useful?"
- "How does semantic search differ from keyword search?"
- "Explain retrieval augmented generation."

## Tech Stack
- Python 3.9+
- FastAPI
- Streamlit
- sentence-transformers
- OpenAI (`gpt-3.5-turbo`)
- Endee Vector DB (local service)
- msgpack

## Notes
- Set `OPENAI_API_KEY` environment variable to enable real LLM responses.
- Without an API key, the app returns contextual summary fallback.
