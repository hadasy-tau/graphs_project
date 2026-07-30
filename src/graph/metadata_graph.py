"""Metadata graph: connect documents by similarity of their bibliographic record."""
from __future__ import annotations

import logging

import torch

from src.graph.base import GraphBuilder
from src.graph.mutual_knn import cosine_mutual_knn_mask, load_shared_mutual_knn_k

logger = logging.getLogger(__name__)

_UNSET = object()


class MetadataGraphBuilder(GraphBuilder):
    """Add edge (i, j) when i and j are mutual k-NN by metadata-record similarity."""

    def __init__(
        self,
        record_embeddings: torch.Tensor,
        mutual_knn_k: int | None | object = _UNSET,
    ):
        if mutual_knn_k is _UNSET:
            mutual_knn_k = load_shared_mutual_knn_k()
        if mutual_knn_k is None:
            raise ValueError(
                "MetadataGraphBuilder requires mutual_knn_k "
                "(set it explicitly or as top-level mutual_knn_k in config)"
            )
        self.record_embeddings = record_embeddings
        self.mutual_knn_k = mutual_knn_k

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        # `embeddings` (document text) is deliberately unused: this graph's
        # edges come from the metadata records instead.
        if len(self.record_embeddings) != len(corpus):
            raise ValueError(
                f"record_embeddings has {len(self.record_embeddings)} rows "
                f"but corpus has {len(corpus)} documents"
            )

        mutual_mask = cosine_mutual_knn_mask(self.record_embeddings, self.mutual_knn_k)

        edges = []
        rows, cols = torch.nonzero(mutual_mask, as_tuple=True)
        for idx in range(len(rows)):
            i, j = rows[idx].item(), cols[idx].item()
            if i >= j:
                continue  # upper triangle only
            title_i = corpus[i]["title"]
            title_j = corpus[j]["title"]
            edge_text = f"Similar metadata record: {title_i} and {title_j}"
            edges.append((i, j, edge_text))

        logger.info(
            "MetadataGraph (mutual k-NN k=%d): %d edges",
            self.mutual_knn_k, len(edges),
        )
        return edges
