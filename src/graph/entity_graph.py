"""Entity-based graph: connect documents sharing named entities."""
from __future__ import annotations

import logging
from collections import defaultdict

import torch

from src.graph.base import GraphBuilder
from src.graph.mutual_knn import mutual_knn_pairs

logger = logging.getLogger(__name__)


class EntityGraphBuilder(GraphBuilder):
    """Add edge (i, j) when docs share enough canonical entities.

    Without mutual_knn_k: keep pairs with >= min_shared_entities (from config).

    With mutual_knn_k: candidates need >= 1 shared entity; each node keeps its
    top-K neighbors by overlap count, and an edge is kept only if reciprocal.
    """

    def __init__(
        self,
        min_shared_entities: int = 1,
        mutual_knn_k: int | None = None,
    ):
        self.min_shared_entities = min_shared_entities
        self.mutual_knn_k = mutual_knn_k

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        # Inverted index: entity → list of doc_ids
        inverted: dict[str, list[int]] = defaultdict(list)
        for doc_id, ent_set in entities.items():
            for ent in ent_set:
                inverted[ent].append(doc_id)

        # Count shared entities per doc pair
        shared: dict[tuple[int, int], set[str]] = defaultdict(set)
        for ent, doc_ids in inverted.items():
            for a in range(len(doc_ids)):
                for b in range(a + 1, len(doc_ids)):
                    i, j = sorted([doc_ids[a], doc_ids[b]])
                    shared[(i, j)].add(ent)

        # No k-NN: use configured min. With k-NN: allow any shared entity (>=1);
        # k controls density.
        min_shared = self.min_shared_entities if self.mutual_knn_k is None else 1
        candidates = {
            (i, j): ents for (i, j), ents in shared.items() if len(ents) >= min_shared
        }

        if self.mutual_knn_k is not None:
            scored = {pair: float(len(ents)) for pair, ents in candidates.items()}
            keep = mutual_knn_pairs(scored, self.mutual_knn_k)
            candidates = {pair: candidates[pair] for pair in keep}

        edges = []
        for (i, j), ents in candidates.items():
            entity_list = ", ".join(sorted(ents)[:10])  # cap for readability
            edge_text = f"Shared entities: {entity_list}"
            edges.append((i, j, edge_text))

        knn_note = f", mutual k-NN k={self.mutual_knn_k}" if self.mutual_knn_k else ""
        logger.info(
            "EntityGraph (min_shared=%d%s): %d edges from %d entity co-occurrences",
            min_shared, knn_note, len(edges), sum(len(v) for v in inverted.values()),
        )
        return edges
