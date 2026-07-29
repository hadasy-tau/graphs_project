"""Semantic similarity graph: connect documents by embedding cosine similarity."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

from src.graph.base import GraphBuilder
from src.graph.mutual_knn import mutual_knn_mask

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "base.yaml"
_UNSET = object()


def _load_mutual_knn_k() -> int | None:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("mutual_knn_k")


class SemanticGraphBuilder(GraphBuilder):
    """Add edge (i, j) when i and j are mutual k-NN by cosine similarity.

    Edge (i,j) is kept only if j is in i's top-K neighbors AND i is in j's
    top-K neighbors. This controls graph density as a confounding variable
    (typically matched to the entity graph's average degree, or set via the
    shared top-level mutual_knn_k config).

    When mutual_knn_k is not provided, the value is read from config/base.yaml.
    """

    def __init__(self, mutual_knn_k: int | None | object = _UNSET):
        if mutual_knn_k is _UNSET:
            mutual_knn_k = _load_mutual_knn_k()
        if mutual_knn_k is None:
            raise ValueError(
                "SemanticGraphBuilder requires mutual_knn_k "
                "(set it explicitly or as top-level mutual_knn_k in config)"
            )
        self.mutual_knn_k = mutual_knn_k

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        # For unit-length vectors cosine similarity is the dot product, so the
        # whole N x N matrix is a single matmul.
        if (embeddings.norm(dim=1) - 1).abs().max() > 1e-3:
            raise ValueError("SemanticGraphBuilder needs unit-normalized embeddings")
        sim = embeddings @ embeddings.T  # (N, N)

        sim.fill_diagonal_(0.0)

        mutual_mask = mutual_knn_mask(sim, self.mutual_knn_k)

        edges = []
        rows, cols = torch.nonzero(mutual_mask, as_tuple=True)
        for idx in range(len(rows)):
            i, j = rows[idx].item(), cols[idx].item()
            if i >= j:
                continue  # upper triangle only
            title_i = corpus[i]["title"]
            title_j = corpus[j]["title"]
            edge_text = f"Semantically related: {title_i} and {title_j}"
            edges.append((i, j, edge_text))

        logger.info(
            "SemanticGraph (mutual k-NN k=%d): %d edges",
            self.mutual_knn_k, len(edges),
        )
        return edges
