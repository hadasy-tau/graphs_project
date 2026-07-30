"""Mutual k-NN sparsification shared by entity / metadata / semantic graphs."""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    """The active config: whatever scripts/experiment.py pointed us at.

    Read at call time, not import time, so a sweep over mutual_knn_k cannot
    silently fall back to config/base.yaml.
    """
    env = os.environ.get("GRAPHS_PROJECT_CONFIG")
    path = Path(env) if env else _ROOT / "config" / "base.yaml"
    return path if path.is_absolute() else _ROOT / path


def load_shared_mutual_knn_k() -> int | None:
    """Top-level mutual_knn_k from the active config; None when not pinned.

    The builders take k as an argument and scripts/02 always passes it; this is
    the fallback for a builder constructed without one.
    """
    with open(_config_path(), encoding="utf-8") as f:
        return yaml.safe_load(f).get("mutual_knn_k")


def mutual_knn_pairs(
    scored_pairs: Mapping[tuple[int, int], float],
    k: int,
) -> set[tuple[int, int]]:
    """Keep undirected pairs that are mutual top-K neighbors by score.

    ``scored_pairs`` keys must be ``(i, j)`` with ``i < j``. An edge is kept
    only if ``j`` is among ``i``'s top-K scored neighbors and vice versa.

    Use this for sparse candidate graphs (entity overlap, metadata matches).
    For a dense similarity matrix, prefer :func:`mutual_knn_mask`.
    """
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, j), score in scored_pairs.items():
        neighbors[i].append((j, score))
        neighbors[j].append((i, score))

    topk: dict[int, set[int]] = {}
    for node, nbrs in neighbors.items():
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)
        topk[node] = {n for n, _ in nbrs_sorted[:k]}

    keep: set[tuple[int, int]] = set()
    for (i, j) in scored_pairs:
        if j in topk.get(i, ()) and i in topk.get(j, ()):
            keep.add((i, j))
    return keep


def mutual_knn_mask(sim: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mutual top-K mask from a dense similarity / score matrix.

    ``sim[i, j]`` is the directed score from i to j. Diagonal should already
    be zeroed (no self-loops). Returns a symmetric bool matrix.
    """
    n = sim.shape[0]
    k = min(k, n - 1)
    topk_mask = torch.zeros_like(sim, dtype=torch.bool)
    _, topk_idx = torch.topk(sim, k, dim=1)
    topk_mask.scatter_(1, topk_idx, True)
    return topk_mask & topk_mask.t()


def cosine_mutual_knn_mask(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """Mutual top-K mask by cosine similarity over unit-normalized embeddings.

    For unit-length vectors cosine similarity is the dot product, so the whole
    N x N matrix is a single matmul. Used by any graph whose edges come from
    embedding similarity, whatever text those embeddings were built from
    (document body, metadata record, ...).
    """
    if (embeddings.norm(dim=1) - 1).abs().max() > 1e-3:
        raise ValueError("cosine_mutual_knn_mask needs unit-normalized embeddings")
    sim = embeddings @ embeddings.T  # (N, N)
    sim.fill_diagonal_(0.0)  # no self-loops
    return mutual_knn_mask(sim, k)
