from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert a list of texts to vector embeddings."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Convert numpy arrays to python lists for JSON serialization
    return [emb.tolist() for emb in embeddings]
