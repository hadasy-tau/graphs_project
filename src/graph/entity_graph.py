"""Entity-based graph: connect documents sharing named entities."""
from __future__ import annotations

import logging
from collections import defaultdict

import torch

from src.graph.base import GraphBuilder

logger = logging.getLogger(__name__)


class EntityGraphBuilder(GraphBuilder):
    """Add edge (i, j) when docs share >= min_shared_entities canonical entities."""

    def __init__(self, min_shared_entities: int = 1):
        self.min_shared_entities = min_shared_entities

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

        edges = []
        for (i, j), ents in shared.items():
            if len(ents) >= self.min_shared_entities:
                entity_list = ", ".join(sorted(ents)[:10])  # cap for readability
                edge_text = f"Shared entities: {entity_list}"
                edges.append((i, j, edge_text))

        logger.info(
            "EntityGraph (min_shared=%d): %d edges from %d entity co-occurrences",
            self.min_shared_entities, len(edges), sum(len(v) for v in inverted.values()),
        )
        return edges
