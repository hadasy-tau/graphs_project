"""Dense (no-graph) and graph-neighborhood retrieval baselines."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def precompute_similarities(query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
    """Batch cosine similarity: (Q, D) @ (D, N) -> (Q, N).

    Call once before the query loop; pass rows into retrieve_dense /
    retrieve_graph_neighborhood to avoid redundant per-query dot-products.
    """
    q = F.normalize(query_embs, dim=-1)
    d = F.normalize(doc_embs, dim=-1)
    return q @ d.T


def build_adj(graph_data: Data) -> dict[int, list[int]]:
    """Build undirected adjacency list from edge_index.

    Call once per graph and pass into retrieve_graph_neighborhood so the
    O(E) list construction is not repeated for every query.
    """
    adj: dict[int, list[int]] = {}
    for src, dst in graph_data.edge_index.t().tolist():
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)
    return adj


def retrieve_dense(sims_row: torch.Tensor, k: int) -> list[int]:
    """Return top-K doc indices from a precomputed similarity row."""
    k = min(k, sims_row.shape[0])
    _, topk_idx = torch.topk(sims_row, k)
    return topk_idx.tolist()


def retrieve_graph_neighborhood(
    sims_row: torch.Tensor,
    graph_data: Data,
    k: int,
    expand_hops: int = 1,
    adj: dict[int, list[int]] | None = None,
    rerank_to: int | None = None,
) -> list[int]:
    """Top-K by precomputed sim, expand expand_hops graph hops, optionally rerank.

    Pass a prebuilt adj (from build_adj) to avoid rebuilding per query.
    If rerank_to is set and the expanded set exceeds it, re-score all candidates
    by cosine similarity to the query and keep only the top rerank_to docs.
    This recovers precision after the expansion step without sacrificing recall.
    """
    k = min(k, sims_row.shape[0])
    _, topk_idx = torch.topk(sims_row, k)
    seed_nodes = set(topk_idx.tolist())

    if graph_data.edge_index.numel() == 0 or expand_hops == 0:
        candidates = sorted(seed_nodes)
    else:
        if adj is None:
            adj = build_adj(graph_data)

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

        candidates = sorted(visited)

    if rerank_to is not None and len(candidates) > rerank_to:
        cand_tensor = torch.tensor(candidates, dtype=torch.long)
        scores = sims_row[cand_tensor]
        top_idx = scores.topk(rerank_to).indices
        return sorted(cand_tensor[top_idx].tolist())

    return candidates
