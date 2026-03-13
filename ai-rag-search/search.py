import json
import logging
from typing import List, Dict

import msgpack
import requests

from .embed import embed_texts

ENDEE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "docs_index"


def create_index(dim: int = 384, space_type: str = "cosine") -> bool:
    """Create [or ignore existing] index in Endee."""
    payload = {"index_name": INDEX_NAME, "dim": dim, "space_type": space_type}
    try:
        resp = requests.post(f"{ENDEE_URL}/index/create", json=payload, timeout=15)
        if resp.status_code == 200:
            return True
        # Existing index may return 400/409 in some versions, treat as success on known duplicate message
        if resp.status_code in (400, 409) and "already exists" in resp.text.lower():
            return True
        logging.warning("Create index failed %s: %s", resp.status_code, resp.text)
        return False
    except requests.RequestException as exc:
        logging.error("Error creating index: %s", exc)
        return False


def upsert_documents(texts: List[str]) -> Dict[str, int]:
    """Embed and insert documents into the Endee index."""
    if not texts:
        return {"inserted": 0}

    loaded = create_index()
    if not loaded:
        raise RuntimeError("Could not create or access Endee index")

    embeddings = embed_texts(texts)
    assert len(embeddings) == len(texts)

    batch = []
    for idx, (text, vector) in enumerate(zip(texts, embeddings), start=1):
        batch.append({
            "id": str(idx),
            # Use raw text as metadata; Endee stores it as bytes in `meta`.
            "meta": text,
            "vector": vector,
        })

    try:
        resp = requests.post(f"{ENDEE_URL}/index/{INDEX_NAME}/vector/insert", json=batch, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Endee insert failed [{resp.status_code}]: {resp.text}")
        return {"inserted": len(batch)}
    except requests.RequestException as exc:
        raise RuntimeError(f"Endee insert request failed: {exc}") from exc


def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Do a semantic search in Endee and return text payloads."""
    if not query:
        return []

    query_emb = embed_texts([query])[0]
    payload = {"k": top_k, "vector": query_emb, "include_vectors": False}

    try:
        resp = requests.post(f"{ENDEE_URL}/index/{INDEX_NAME}/search", json=payload, timeout=20)
        resp.raise_for_status()

        # Search response is msgpack type. Parse accordingly.
        decoded = msgpack.unpackb(resp.content, raw=False)
        # Expect decoded is a dict: {"results":[{...}]}
        results = decoded.get("results", []) if isinstance(decoded, dict) else []

        parsed = []
        for item in results:
            item_meta = item.get("meta")
            decoded_meta = ""
            if isinstance(item_meta, (bytes, bytearray)):
                decoded_meta = item_meta.decode("utf-8", errors="replace")
            elif isinstance(item_meta, str):
                decoded_meta = item_meta
            else:
                # Sometimes metadata may be encoded JSON: handle gracefully
                decoded_meta = json.dumps(item_meta, ensure_ascii=False)

            parsed.append({
                "id": item.get("id"),
                "score": item.get("similarity"),
                "text": decoded_meta,
            })

        return parsed

    except requests.RequestException as exc:
        raise RuntimeError(f"Endee search request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not parse Endee search response: {exc}") from exc
