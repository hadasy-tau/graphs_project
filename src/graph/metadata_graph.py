"""Metadata-based graph: connect documents that share at least 2 metadata fields."""
from __future__ import annotations

import logging

import torch

from src.graph.base import GraphBuilder

logger = logging.getLogger(__name__)


class MetadataGraphBuilder(GraphBuilder):
    """Add edge (i, j) when docs share at least 2 of: category, source, author."""

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        n = len(corpus)
        edges = []

        for i in range(n):
            for j in range(i + 1, n):
                matching = _matching_fields(corpus[i], corpus[j])
                if matching:
                    edge_text = "Same " + ", ".join(matching)
                    edges.append((i, j, edge_text))

        logger.info("MetadataGraph: %d edges", len(edges))
        return edges


def _matching_fields(doc_i: dict, doc_j: dict) -> list[str]:
    matches = []

    cat_i = (doc_i.get("category") or "").strip().lower()
    cat_j = (doc_j.get("category") or "").strip().lower()
    if cat_i and cat_j and cat_i == cat_j:
        matches.append(f"category: {cat_i}")

    src_i = (doc_i.get("source") or "").strip().lower()
    src_j = (doc_j.get("source") or "").strip().lower()
    if src_i and src_j and src_i == src_j:
        matches.append(f"source: {src_i}")

    auth_i = (doc_i.get("author") or "").strip().lower()
    auth_j = (doc_j.get("author") or "").strip().lower()
    if auth_i and auth_j and auth_i == auth_j:
        matches.append(f"author: {auth_i}")

    return matches if len(matches) >= 2 else []
