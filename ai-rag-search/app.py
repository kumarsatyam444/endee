import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag_pipeline import generate_answer
from .search import upsert_documents

app = FastAPI(title="Endee AI RAG Search")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "docs.txt"


class AskRequest(BaseModel):
    question: str


@app.post("/upload-documents")
def upload_documents():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_FILE}")

    content = DATA_FILE.read_text(encoding="utf-8").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Source document file is empty")

    # Split into paragraphs. Snippet separators by blank line are accepted.
    docs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not docs:
        raise HTTPException(status_code=400, detail="No documents found in docs.txt")

    try:
        result = upsert_documents(docs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "success", "documents": len(docs), "details": result}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        answer = generate_answer(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def healthcheck():
    return {"status": "healthy"}


def streamlit_ui():
    import streamlit as st

    st.set_page_config(page_title="Endee RAG Semantic Search", layout="centered")
    st.title("Endee RAG Semantic Search Demo")
    st.write("Type a question and get a RAG answer generated with Endee vector search.")

    endpoint = st.text_input("Backend URL", value="http://localhost:8000")
    query = st.text_area("Your question", height=120)

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying Endee and generating answer..."):
                try:
                    resp = requests.post(f"{endpoint}/ask", json={"question": query}, timeout=120)
                    if resp.status_code != 200:
                        st.error(f"Failed: {resp.status_code} - {resp.text}")
                    else:
                        data = resp.json()
                        st.subheader("Answer")
                        st.write(data.get("answer", "No answer returned"))
                except requests.RequestException as exc:
                    st.error(f"Connection failed to backend: {exc}")


if __name__ == "__main__":
    import sys

    if "streamlit" in " ".join(sys.argv).lower():
        streamlit_ui()
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
