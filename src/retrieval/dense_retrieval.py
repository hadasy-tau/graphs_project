"""Dense (no-graph) and graph-neighborhood retrieval baselines."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def retrieve_dense(
    query_emb: torch.Tensor,
    doc_embs: torch.Tensor,
    k: int,
) -> list[int]:
    """Return top-K doc indices by cosine similarity (no graph)."""
    sims = F.cosine_similarity(query_emb.unsqueeze(0), doc_embs, dim=-1)
    k = min(k, doc_embs.shape[0])
    _, topk_idx = torch.topk(sims, k)
    return topk_idx.tolist()


def retrieve_graph_neighborhood(
    query_emb: torch.Tensor,
    doc_embs: torch.Tensor,
    graph_data: Data,
    k: int,
    expand_hops: int = 1,
) -> list[int]:
    """Top-K by cosine sim, then expand expand_hops graph hops.

    Used as the no-PCST condition: isolates the effect of PCST optimization
    from the graph structure itself.
    """
    sims = F.cosine_similarity(query_emb.unsqueeze(0), doc_embs, dim=-1)
    k = min(k, doc_embs.shape[0])
    _, topk_idx = torch.topk(sims, k)
    seed_nodes = set(topk_idx.tolist())

    if graph_data.edge_index.numel() == 0 or expand_hops == 0:
        return sorted(seed_nodes)

    # Build adjacency list
    edge_index = graph_data.edge_index
    adj: dict[int, list[int]] = {}
    for src, dst in edge_index.t().tolist():
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)

    frontier = set(seed_nodes)
    visited = set(seed_nodes)
    for _ in range(expand_hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        visited |= next_frontier
        frontier = next_frontier

    return sorted(visited)
