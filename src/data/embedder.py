"""Sentence embedding utilities for documents and edge texts."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    device: str | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Embed a list of strings and return a (N, D) float32 tensor.

    normalize=True produces unit-norm vectors, enabling cosine similarity
    via simple dot product.
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_tensor=True,
    )
    return embeddings.float().cpu()


def embed_documents(
    docs: list[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    device: str | None = None,
    max_tokens: int = 512,
) -> torch.Tensor:
    """Embed documents as 'title [SEP] body[:max_tokens]'."""
    texts = []
    for doc in docs:
        body_tokens = doc["body"].split()[:max_tokens]
        body = " ".join(body_tokens)
        texts.append(f"{doc['title']} [SEP] {body}")
    return embed_texts(texts, model_name=model_name, batch_size=batch_size, device=device)


def embed_queries(
    queries: list[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    device: str | None = None,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
) -> torch.Tensor:
    """Embed query strings with an optional retrieval instruction prefix.

    For BGE retrieval models, the prefix is applied to queries only.
    Documents remain unprefixed.
    """
    texts = [query_prefix + q["query"] for q in queries]
    return embed_texts(texts, model_name=model_name, batch_size=batch_size, device=device)


def load_or_compute(
    path: str | Path,
    compute_fn,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Load tensor from disk if it exists, else compute and save."""
    path = Path(path)
    if path.exists():
        logger.info("Loading cached embeddings from %s", path)
        return torch.load(path, weights_only=True)
    logger.info("Computing embeddings → %s", path)
    tensor = compute_fn(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return tensor
