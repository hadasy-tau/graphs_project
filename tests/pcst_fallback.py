"""A stand-in for the pcst_fast package, for machines that cannot install it.

You should not need to read this file. It is a compatibility shim, not part of
the project's method.

WHY IT EXISTS
-------------
PyG's retrieval_via_pcst (the G-Retriever subgraph selector this project uses)
does `from pcst_fast import pcst_fast`. The pcst-fast package ships C++ source
with no prebuilt wheels, so installing it needs a compiler - on Windows,
Microsoft C++ Build Tools 14.0+. Without it the most interesting stage of the
pipeline could not run at all.

WHAT IT IS NOT
--------------
The real pcst_fast solves Prize-Collecting Steiner Tree with the
Goemans-Williamson approximation. This is a plain greedy heuristic: start at the
highest-prize node, then keep attaching whichever node's prize most exceeds the
cost of reaching it. It returns a plausible connected subgraph, not the same
one. Fine for watching the pipeline work; not fine for reporting numbers.

Install the real solver before running scripts/03_run_retrieval.py:

    pip install pcst-fast
"""
from __future__ import annotations

import heapq
import logging
import sys
from math import inf

import numpy as np

logger = logging.getLogger(__name__)

IS_FALLBACK = True  # lets callers tell this shim apart from the real package


def install_if_missing() -> str:
    """Make `import pcst_fast` work. Returns "real" or "fallback"."""
    try:
        import pcst_fast
        return "fallback" if getattr(pcst_fast, "IS_FALLBACK", False) else "real"
    except ImportError:
        sys.modules["pcst_fast"] = sys.modules[__name__]
        logger.warning(
            "pcst-fast is not installed, using the approximate fallback in "
            "tests/pcst_fallback.py. The pipeline will run, but PCST output is "
            "NOT the real solver's - do not report these numbers."
        )
        return "fallback"


def pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level):
    """Greedy prize-collecting Steiner tree. Same signature as the C++ version.

    Args:
        edges: (E, 2) array of endpoint pairs.
        prizes: (V,) node prizes. Its length defines the number of nodes.
        costs: (E,) edge costs.
        root: node to start from, or -1 to start from the highest-prize node.
        num_clusters, pruning, verbosity_level: accepted for signature
            compatibility; only num_clusters=1 (what this project uses) is
            meaningful here.

    Returns:
        (selected_node_indices, selected_edge_indices), both int64 arrays.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n_nodes = len(prizes)

    if n_nodes == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # node -> list of (neighbour, edge cost, edge id)
    adjacency = [[] for _ in range(n_nodes)]
    for edge_id, (u, v) in enumerate(edges):
        adjacency[u].append((int(v), float(costs[edge_id]), edge_id))
        adjacency[v].append((int(u), float(costs[edge_id]), edge_id))

    start = int(root) if root >= 0 else int(np.argmax(prizes))
    tree_nodes = {start}
    tree_edges = set()

    while True:
        # cheapest path from anywhere in the tree to every other node
        distance, via_edge, via_node = _dijkstra_from_set(adjacency, tree_nodes, n_nodes)

        # the node whose prize most exceeds the cost of reaching it
        best_gain, best_node = 0.0, None
        for node in range(n_nodes):
            if node in tree_nodes or distance[node] == inf:
                continue
            gain = prizes[node] - distance[node]
            if gain > best_gain:
                best_gain, best_node = gain, node

        if best_node is None:  # nothing left worth connecting
            break

        # absorb the whole path back to the tree
        node = best_node
        while node not in tree_nodes:
            tree_edges.add(via_edge[node])
            tree_nodes.add(node)
            node = via_node[node]

    return (np.array(sorted(tree_nodes), dtype=np.int64),
            np.array(sorted(tree_edges), dtype=np.int64))


def _dijkstra_from_set(adjacency, sources, n_nodes):
    """Shortest path cost from a SET of start nodes to every node."""
    distance = [inf] * n_nodes
    via_edge = [-1] * n_nodes
    via_node = [-1] * n_nodes

    heap = [(0.0, s) for s in sources]
    for s in sources:
        distance[s] = 0.0
    heapq.heapify(heap)

    while heap:
        d, node = heapq.heappop(heap)
        if d > distance[node]:
            continue
        for neighbour, cost, edge_id in adjacency[node]:
            if d + cost < distance[neighbour]:
                distance[neighbour] = d + cost
                via_edge[neighbour] = edge_id
                via_node[neighbour] = node
                heapq.heappush(heap, (distance[neighbour], neighbour))

    return distance, via_edge, via_node
