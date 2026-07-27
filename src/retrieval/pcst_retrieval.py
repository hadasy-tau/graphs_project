"""PCST-based subgraph retrieval using G-Retriever's implementation from PyG."""
from __future__ import annotations

import logging

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def retrieve_with_pcst(
    query_emb: torch.Tensor,
    graph_data: Data,
    textual_nodes,
    textual_edges,
    topk: int = 3,
    topk_e: int = 5,
    cost_e: float = 0.5,
) -> list[int]:
    """Return doc_ids selected by PCST for a single query.

    Uses torch_geometric.llm.utils.backend_utils.retrieval_via_pcst directly.
    Assigns node prizes by top-K cosine similarity and edge prizes by cosine
    similarity between the query and embedded edge descriptions.

    Args:
        query_emb: (D,) query embedding, unit-normalized.
        graph_data: PyG Data with x, edge_index, edge_attr, node_idx, edge_idx.
        textual_nodes: pd.DataFrame with columns node_id, node_attr (N rows).
        textual_edges: pd.DataFrame with columns src, edge_attr, dst (E rows).
        topk: number of top nodes to assign prizes.
        topk_e: number of top edges to assign prizes.
        cost_e: base edge traversal cost.

    Returns:
        List of doc_ids in the retrieved subgraph.
    """
    from torch_geometric.llm.utils.backend_utils import retrieval_via_pcst

    # retrieval_via_pcst expects q_emb shape [1, D] or [D]
    subgraph, _desc = retrieval_via_pcst(
        data=graph_data,
        q_emb=query_emb,
        textual_nodes=textual_nodes,
        textual_edges=textual_edges,
        topk=topk,
        topk_e=topk_e,
        cost_e=cost_e,
    )

    node_indices = subgraph.node_idx
    if hasattr(node_indices, "tolist"):
        return node_indices.tolist()
    return list(node_indices)
