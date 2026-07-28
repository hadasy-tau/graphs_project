"""Semantic similarity graph: connect documents by embedding cosine similarity."""
from __future__ import annotations

import logging

import torch

from src.graph.base import GraphBuilder

logger = logging.getLogger(__name__)


class SemanticGraphBuilder(GraphBuilder):
    """Add edge (i, j) when cosine_sim(emb_i, emb_j) >= threshold.

    Applies mutual k-NN sparsification to control for graph density as a
    confounding variable: edge (i,j) is kept only if j is in i's top-K
    neighbors AND i is in j's top-K neighbors.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        mutual_knn_k: int | None = None,
    ):
        self.threshold = threshold
        self.mutual_knn_k = mutual_knn_k  # None = no mutual k-NN constraint

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        n = embeddings.shape[0]

        # For unit-length vectors cosine similarity is the dot product, so the
        # whole N x N matrix is a single matmul.
        if (embeddings.norm(dim=1) - 1).abs().max() > 1e-3:
            raise ValueError("SemanticGraphBuilder needs unit-normalized embeddings")
        sim = embeddings @ embeddings.T  # (N, N)

        # Zero diagonal (no self-loops) and threshold
        sim.fill_diagonal_(0.0)
        above = sim >= self.threshold

        # Mutual k-NN sparsification
        if self.mutual_knn_k is not None:
            k = min(self.mutual_knn_k, n - 1)
            topk_mask = torch.zeros_like(sim, dtype=torch.bool)
            _, topk_idx = torch.topk(sim, k, dim=1)
            topk_mask.scatter_(1, topk_idx, True)
            mutual_mask = topk_mask & topk_mask.t()
            above = above & mutual_mask

        edges = []
        rows, cols = torch.nonzero(above, as_tuple=True)
        for idx in range(len(rows)):
            i, j = rows[idx].item(), cols[idx].item()
            if i >= j:
                continue  # upper triangle only
            title_i = corpus[i]["title"]
            title_j = corpus[j]["title"]
            edge_text = f"Semantically related: {title_i} and {title_j}"
            edges.append((i, j, edge_text))

        knn_note = f", mutual k-NN k={self.mutual_knn_k}" if self.mutual_knn_k else ""
        logger.info(
            "SemanticGraph (threshold=%.2f%s): %d edges",
            self.threshold, knn_note, len(edges),
        )
        return edges
