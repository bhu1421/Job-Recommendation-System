"""Optional dense embeddings used for meaning-based job matching."""

from __future__ import annotations

import numpy as np
from functools import lru_cache


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic matching requires the optional dependency. Install "
            "requirements-semantic.txt and rebuild with --semantic."
        ) from error
    return SentenceTransformer(model_name)


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode text as normalized vectors; requires the optional package."""
    model = _load_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)
