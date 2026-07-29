"""Combined graph: union of entity, metadata, and semantic edges.

Each sub-builder gets the same corpus and document embeddings; the metadata
builder additionally carries its own record embeddings internally.
"""
from __future__ import annotations

import logging

import torch

from src.graph.base import GraphBuilder
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.metadata_graph import MetadataGraphBuilder
from src.graph.semantic_graph import SemanticGraphBuilder

logger = logging.getLogger(__name__)


class CombinedGraphBuilder(GraphBuilder):
    """Union of entity + metadata + semantic edges.

    For edges that appear in multiple strategies, the edge text concatenates
    all contributing descriptions.
    """

    def __init__(
        self,
        entity_builder: EntityGraphBuilder | None = None,
        metadata_builder: MetadataGraphBuilder | None = None,
        semantic_builder: SemanticGraphBuilder | None = None,
    ):
        # metadata_builder has no usable default: it needs the metadata record
        # embeddings, which only the caller can supply.
        if metadata_builder is None:
            raise ValueError("CombinedGraphBuilder requires a metadata_builder")
        self.entity_builder = entity_builder or EntityGraphBuilder()
        self.metadata_builder = metadata_builder
        self.semantic_builder = semantic_builder or SemanticGraphBuilder()

    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        # Gather edges from all three strategies
        all_edges: dict[tuple[int, int], list[str]] = {}

        for builder, label in [
            (self.entity_builder, "entity"),
            (self.metadata_builder, "metadata"),
            (self.semantic_builder, "semantic"),
        ]:
            for i, j, edge_text in builder.get_edges(corpus, embeddings, entities):
                key = (min(i, j), max(i, j))
                if key not in all_edges:
                    all_edges[key] = []
                all_edges[key].append(f"[{label}] {edge_text}")

        edges = []
        for (i, j), texts in all_edges.items():
            combined_text = "; ".join(texts)
            edges.append((i, j, combined_text))

        logger.info("CombinedGraph: %d edges (union of entity + metadata + semantic)", len(edges))
        return edges
