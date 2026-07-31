"""Random baseline graphs that isolate graph TYPE from graph STRUCTURE.

Given an already-built graph (entity / metadata / semantic / combined, saved
by scripts/02_build_graphs.py as ``<name>_graph.pt`` + its two textual CSVs),
this module derives two null-model variants that share its node set:

1. **random-structure graph** (:func:`random_structure_graph`) — same nodes,
   same edge count/avg degree, but edges connect uniformly random node pairs
   instead of whatever rule built the original graph: "this graph's size,
   with none of its structure".

2. **shuffled-nodes graph** (:func:`shuffled_nodes_graph`) — identical edge
   topology (every structural statistic matches the original exactly), but
   node CONTENT (embedding + text) is randomly permuted across positions:
   the true content <-> structure correspondence is destroyed while the
   structure itself is untouched.

Run it by hand against graphs already saved under data/graphs/:

    python src/graph/random_baselines.py --graph entity --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPHS_DIR = ROOT / "data" / "graphs"


def _unique_forward_edges(data: Data) -> torch.Tensor:
    """Row indices of the ``src < dst`` half of an undirected edge_index.

    GraphBuilder.build() always stores both directions of every undirected
    edge (see src/graph/base.py); the ``src < dst`` half is exactly one copy
    of each edge, giving the true undirected edge count and letting us pull
    out each edge's (attr, text) pair exactly once.
    """
    return (data.edge_index[0] < data.edge_index[1]).nonzero(as_tuple=True)[0]


def random_structure_graph(
    data: Data,
    textual_nodes: pd.DataFrame,
    textual_edges: pd.DataFrame,
    seed: int | None = None,
) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    """Erdos-Renyi G(n, m) baseline: same nodes, same edge count/avg degree.

    Node set (x / textual_nodes) is untouched. The m undirected edges are
    rewired to uniformly random node pairs; each new edge recycles one of the
    original graph's own (edge_attr vector, edge text) pairs so both graphs
    draw edge content from the same distribution and no embedding model needs
    to be re-run — only which nodes each piece of content connects changes.
    """
    rng = random.Random(seed)
    n = data.num_nodes

    fwd = _unique_forward_edges(data)
    m = int(fwd.numel())
    max_possible = n * (n - 1) // 2
    if m > max_possible:
        raise ValueError(f"{m} edges requested but only {max_possible} possible among {n} nodes")

    # Sample m distinct unordered pairs uniformly at random.
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < m:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.add((i, j) if i < j else (j, i))
    pair_list = list(pairs)
    rng.shuffle(pair_list)

    # Recycle the original edges' own content, reshuffled onto the new pairs.
    order = list(range(m))
    rng.shuffle(order)
    attr_pool = data.edge_attr[fwd][order] if data.edge_attr is not None else None
    text_pool = textual_edges["edge_attr"].to_numpy()[fwd.numpy()][order]

    srcs = [p[0] for p in pair_list]
    dsts = [p[1] for p in pair_list]
    all_srcs = srcs + dsts
    all_dsts = dsts + srcs

    new_edge_index = (
        torch.tensor([all_srcs, all_dsts], dtype=torch.long) if m > 0 else torch.zeros((2, 0), dtype=torch.long)
    )
    if attr_pool is not None:
        new_edge_attr = torch.cat([attr_pool, attr_pool], dim=0) if m > 0 else data.edge_attr.new_zeros((0, data.edge_attr.shape[1]))
    else:
        new_edge_attr = None

    new_data = Data(
        x=data.x.clone() if data.x is not None else None,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        node_idx=list(range(n)),
        edge_idx=list(range(2 * m)),
    )

    new_textual_nodes = textual_nodes.copy()

    node_text = textual_nodes["node_attr"].tolist()
    new_textual_edges = pd.DataFrame({
        "src": [node_text[i] for i in all_srcs],
        "edge_attr": list(text_pool) * 2,
        "dst": [node_text[i] for i in all_dsts],
    })

    logger.info(
        "random_structure_graph: n=%d m=%d avg_degree=%.3f (matches input exactly, topology is uniform random)",
        n, m, (2 * m / n) if n else 0.0,
    )
    return new_data, new_textual_nodes, new_textual_edges


def shuffled_nodes_graph(
    data: Data,
    textual_nodes: pd.DataFrame,
    textual_edges: pd.DataFrame,
    seed: int | None = None,
) -> tuple[Data, pd.DataFrame, pd.DataFrame, list[int]]:
    """Node-relabeling baseline implemented by permuting edges, not node content.

    This baseline destroys the real correspondence between graph structure and
    document identity while preserving the invariant that node position equals
    document id. A random permutation rewires every edge (u, v) to
    (perm[u], perm[v]).

    Compared with permuting ``data.x`` directly, this formulation avoids any
    need to map retrieved node positions back to original document ids during
    evaluation. Retrieval can keep returning ordinary document ids.

    The document-level null graph is equivalent to the node-content permutation:
    the same document pairs become connected, edge embeddings stay attached to
    the same edge rows, and node embeddings remain at their original document ids.

    Returns ``(data, textual_nodes, textual_edges, perm)`` where ``perm`` is the
    node permutation used to rewire the edges.
    """
    rng = random.Random(seed)
    n = data.num_nodes
    perm = list(range(n))
    rng.shuffle(perm)

    perm_t = torch.tensor(perm, dtype=torch.long)

    # Rewire edges through the permutation:
    # original edge (u, v) becomes (perm[u], perm[v]).
    new_edge_index = perm_t[data.edge_index]

    new_data = Data(
        x=data.x.clone() if data.x is not None else None,
        edge_index=new_edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
        node_idx=list(range(n)),
        edge_idx=list(range(data.edge_index.shape[1])),
    )

    # Nodes stay in their original document-id positions.
    new_textual_nodes = textual_nodes.copy()

    # Edge descriptions stay with their original edge rows, but endpoint names
    # must be recomputed because the endpoints changed.
    node_text = textual_nodes["node_attr"].tolist()
    new_textual_edges = textual_edges.copy()
    new_textual_edges["src"] = [node_text[i] for i in new_edge_index[0].tolist()]
    new_textual_edges["dst"] = [node_text[i] for i in new_edge_index[1].tolist()]

    logger.info(
        "shuffled_nodes_graph: n=%d, edge topology relabelled through node permutation, "
        "node content/id positions unchanged (seed=%s)",
        n, seed,
    )
    return new_data, new_textual_nodes, new_textual_edges, perm


# ---------------------------------------------------------------------------
# I/O + CLI
# ---------------------------------------------------------------------------

def load_graph(name: str, graphs_dir: Path) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    data = torch.load(graphs_dir / f"{name}_graph.pt", weights_only=False)
    tn = pd.read_csv(graphs_dir / f"{name}_textual_nodes.csv")
    te = pd.read_csv(graphs_dir / f"{name}_textual_edges.csv")
    return data, tn, te


def save_graph(prefix: str, data: Data, tn: pd.DataFrame, te: pd.DataFrame, graphs_dir: Path) -> None:
    torch.save(data, graphs_dir / f"{prefix}_graph.pt")
    tn.to_csv(graphs_dir / f"{prefix}_textual_nodes.csv", index=False)
    te.to_csv(graphs_dir / f"{prefix}_textual_edges.csv", index=False)


def generate_random_baselines(
    name: str,
    graphs_dir: Path = DEFAULT_GRAPHS_DIR,
    seed: int | None = 42,
) -> dict:
    """Build both baselines for one saved graph and write them next to it.

    Reads ``<graphs_dir>/<name>_graph.pt`` (+ its two textual CSVs) and writes:

    - ``<name>_random_structure_graph.pt`` / ``_textual_nodes.csv`` / ``_textual_edges.csv``
    - ``<name>_shuffled_nodes_graph.pt`` / ``_textual_nodes.csv`` / ``_textual_edges.csv``
    - ``<name>_shuffled_nodes_permutation.json`` (the node permutation, for traceability)
    """
    data, tn, te = load_graph(name, graphs_dir)

    rs_data, rs_tn, rs_te = random_structure_graph(data, tn, te, seed=seed)
    save_graph(f"{name}_random_structure", rs_data, rs_tn, rs_te, graphs_dir)

    sn_data, sn_tn, sn_te, perm = shuffled_nodes_graph(data, tn, te, seed=seed)
    save_graph(f"{name}_shuffled_nodes", sn_data, sn_tn, sn_te, graphs_dir)
    (graphs_dir / f"{name}_shuffled_nodes_permutation.json").write_text(json.dumps(perm))

    logger.info("Saved random baselines for '%s' to %s", name, graphs_dir)
    return {
        "random_structure": (rs_data, rs_tn, rs_te),
        "shuffled_nodes": (sn_data, sn_tn, sn_te, perm),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--graph", nargs="+", default=["entity", "metadata", "semantic", "combined"],
        help="graph name(s) to build baselines for, matching data/graphs/<name>_graph.pt "
             "(default: the four standard graphs)",
    )
    p.add_argument(
        "--graphs-dir", type=Path, default=DEFAULT_GRAPHS_DIR,
        help="directory with *_graph.pt / *_textual_*.csv (default: data/graphs)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = p.parse_args()

    for name in args.graph:
        graph_path = args.graphs_dir / f"{name}_graph.pt"
        if not graph_path.exists():
            logger.warning("Skipping '%s': %s not found", name, graph_path)
            continue
        logger.info("Building random baselines for '%s'...", name)
        generate_random_baselines(name, args.graphs_dir, seed=args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
