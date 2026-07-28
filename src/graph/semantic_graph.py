"""Semantic similarity graph: connect documents by embedding cosine similarity."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.graph.base import GraphBuilder

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "base.yaml"
_UNSET = object()


def _load_sem_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["semantic_graph"]


class SemanticGraphBuilder(GraphBuilder):
    """Add edge (i, j) when cosine_sim(emb_i, emb_j) >= threshold.

    Applies mutual k-NN sparsification to control for graph density as a
    confounding variable: edge (i,j) is kept only if j is in i's top-K
    neighbors AND i is in j's top-K neighbors.

    When threshold or mutual_knn_k are not provided, values are read from
    config/base.yaml so there is no hardcoded fallback.
    """

    def __init__(
        self,
        threshold: float | object = _UNSET,
        mutual_knn_k: int | None | object = _UNSET,
    ):
        if threshold is _UNSET or mutual_knn_k is _UNSET:
            sem_cfg = _load_sem_cfg()
            if threshold is _UNSET:
                threshold = sem_cfg["threshold"]
            if mutual_knn_k is _UNSET:
                mutual_knn_k = sem_cfg.get("mutual_knn_k")
        self.threshold = threshold
        self.mutual_knn_k = mutual_knn_k  # None = no mutual k-NN constraint

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        n = embeddings.shape[0]

        # Full N×N cosine similarity matrix (~3MB at float32 for 609 docs)
        sim = F.cosine_similarity(
            embeddings.unsqueeze(1),  # (N, 1, D)
            embeddings.unsqueeze(0),  # (1, N, D)
            dim=-1,
        )  # (N, N)

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
