"""Builds the four graph variants and dumps them to CSV so you can study them.

This is the graph half of the pipeline only - no retrieval, no PCST, no
metrics. Its job is to produce graphs you can open in Excel and argue about.

    python tests/test_build_graphs.py                    # all 609 docs, real models
    python tests/test_build_graphs.py --docs 100         # faster, see the warning below
    python tests/test_build_graphs.py --config config/test_small.yaml --docs 60
    python -m pytest tests/test_build_graphs.py          # quick structural check

Results are written to tests/results/graph_runs/.

WHY THIS IS AFFORDABLE ON THE FULL CORPUS
-----------------------------------------
A graph's structure depends only on the edge RULES. The 1024-dim edge_attr
vectors that GraphBuilder.build() computes are used later by PCST for scoring
and change no edge whatsoever. Embedding tens of thousands of edge texts is the
expensive part of scripts/02, so this test calls get_edges() directly and skips
it. What is left - document embeddings and NER - is minutes, and it is cached.

WHAT --docs COSTS YOU
---------------------
Only the metadata graph survives subsetting intact. Take a subset and:

  metadata   EXACT. The rules are pairwise and absolute, so a subset gives
             precisely the induced subgraph of the full one.
  entity     APPROXIMATE. stop_entity_threshold is a FRACTION of the corpus,
             so on 100 docs it drops entities appearing in >20 of them, while
             on 609 it drops entities appearing in >122. Different entities
             survive, so different edges. Canonicalization also clusters over
             whichever entity vocabulary happens to be present.
  semantic   APPROXIMATE. Pair similarities are absolute and do not move, but
             mutual k-NN is global: your top-6 neighbours among 100 documents
             are not your top-6 among 609. Also k itself is derived from the
             entity graph's density, which just changed.
  combined   APPROXIMATE, since it is the union of the three.

So use --docs to iterate quickly, and the default (whole corpus) for anything
you intend to draw a conclusion from. The test prints which mode it ran in.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                             # so `import src...` works
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` works

import networkx as nx
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from common import SEED, setup_logging, subset_corpus
from src.data.embedder import embed_documents, load_or_compute
from src.data.loader import load_multihop_rag
from src.data.preprocessor import clean_text, extract_entities
from src.graph.combined_graph import CombinedGraphBuilder
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.metadata_graph import MetadataGraphBuilder
from src.graph.semantic_graph import SemanticGraphBuilder

OUT_ROOT = Path(__file__).resolve().parent / "results" / "graph_runs"

# What `pytest` runs: small models, small slice, seconds not minutes.
PYTEST_CONFIG = ROOT / "config" / "test_small.yaml"
PYTEST_DOCS = 60
PYTEST_QUERIES = 15


def build_graphs(config_path, n_docs=None, n_queries=None, seed=SEED):
    """Build the four variants and write them to tests/results/graph_runs/<name>/.

    Args:
        config_path: which YAML to use. config/base.yaml gives the real graphs.
        n_docs: keep only this many documents (None = the whole corpus).
        n_queries: keep only this many queries for the connectivity report
            (None = all of them). Does not affect the graphs themselves.
        seed: which random subset to take.

    Returns a dict of {graph name -> list of (src, dst, edge_text)}.
    """
    setup_logging()
    cfg = yaml.safe_load(open(config_path))

    corpus, queries = load_multihop_rag()
    full_corpus_size = len(corpus)
    corpus, queries = subset_corpus(corpus, queries, n_docs, n_queries, seed)
    exact = len(corpus) == full_corpus_size

    run_name = f"{Path(config_path).stem}_{len(corpus)}docs"
    out = OUT_ROOT / run_name
    (out / "graphs").mkdir(parents=True, exist_ok=True)

    print(f"\nconfig     : {config_path}")
    print(f"corpus     : {len(corpus)} of {full_corpus_size} documents")
    print(f"queries    : {len(queries)}")
    print(f"embeddings : {cfg['embedding']['model']}")
    print(f"NER        : {cfg['ner']['model']}")
    print(f"output     : {out}")
    if exact:
        print("\nFULL CORPUS - these are exactly the graphs config/base.yaml produces\n"
              "(assuming this config IS base.yaml; the models above are what matters).")
    else:
        print(f"\nSUBSET of {len(corpus)}/{full_corpus_size} docs. The metadata graph is exact;\n"
              "entity and semantic (and so combined) are APPROXIMATE - see the module\n"
              "docstring. Do not compare these against a full-corpus run.")

    # --- documents: clean, NER, embed (cached, the slow part) ---
    for doc in corpus:
        doc["body"] = clean_text(doc["body"], max_tokens=cfg["embedding"]["max_tokens"])

    entities = _load_or_extract_entities(corpus, cfg, out / "entities.json")
    doc_embs = load_or_compute(
        out / "embeddings.pt", embed_documents, corpus,
        model_name=cfg["embedding"]["model"], batch_size=cfg["embedding"]["batch_size"],
    )
    print(f"\nembeddings {tuple(doc_embs.shape)}, "
          f"{sum(len(e) for e in entities.values())} entity mentions")

    # --- edges only: get_edges(), never build(), so no edge embedding ---
    entity_builder = EntityGraphBuilder(cfg["entity_graph"]["min_shared_entities"])
    metadata_builder = MetadataGraphBuilder(cfg["metadata_graph"]["date_window_days"])

    # The entity graph goes first because its density sets the semantic graph's
    # k, keeping the two comparable in edge count so only edge MEANING differs.
    entity_edges = entity_builder.get_edges(corpus, doc_embs, entities)
    avg_degree = 2 * len(entity_edges) / len(corpus)
    k = cfg["semantic_graph"]["mutual_knn_k"] or max(1, round(avg_degree))
    print(f"entity avg degree {avg_degree:.2f} -> semantic mutual_knn_k = {k}")

    semantic_builder = SemanticGraphBuilder(cfg["semantic_graph"]["threshold"], mutual_knn_k=k)
    combined_builder = CombinedGraphBuilder(entity_builder, metadata_builder, semantic_builder)

    graphs = {
        "entity": entity_edges,
        "metadata": metadata_builder.get_edges(corpus, doc_embs, entities),
        "semantic": semantic_builder.get_edges(corpus, doc_embs, entities),
        "combined": combined_builder.get_edges(corpus, doc_embs, entities),
    }

    _save(corpus, queries, entities, graphs, out)
    return graphs


def _load_or_extract_entities(corpus, cfg, path):
    """NER is slow, so cache it next to the embeddings."""
    if path.exists():
        print(f"loading cached entities from {path.name}")
        return {int(k): set(v) for k, v in json.load(open(path)).items()}

    entities = extract_entities(
        corpus,
        model_name=cfg["ner"]["model"],
        batch_size=cfg["ner"]["batch_size"],
        keep_types=frozenset(cfg["ner"]["keep_types"]),
        stop_entity_threshold=cfg["ner"]["stop_entity_threshold"],
        canonicalize_threshold=cfg["ner"]["canonicalize_threshold"],
    )
    json.dump({k: sorted(v) for k, v in entities.items()}, open(path, "w"))
    return entities


def _save(corpus, queries, entities, graphs, out):
    """Write everything as CSV, all of it keyed by doc_id."""
    n = len(corpus)
    doc_ids = list(range(n))
    titles = [d["title"] for d in corpus]

    # What is node 7? Entities included so an entity edge traces back to words.
    pd.DataFrame([{
        "doc_id": d["doc_id"],
        "title": d["title"],
        "source": d["source"],
        "category": d["category"],
        "published_at": d["published_at"],
        "n_entities": len(entities[d["doc_id"]]),
        "entities": " | ".join(sorted(entities[d["doc_id"]])),
    } for d in corpus]).to_csv(out / "documents.csv", index=False)

    # The questions, with gold titles spelled out, so you can read a query and
    # see straight away which nodes have to be joined to answer it.
    pd.DataFrame([{
        "query_id": q["query_id"],
        "question_type": q["question_type"],
        "query": q["query"],
        "answer": q["answer"],
        "gold_doc_ids": " ".join(str(g) for g in q["gold_doc_ids"]),
        "gold_titles": " | ".join(titles[g] for g in q["gold_doc_ids"]),
    } for q in queries]).to_csv(out / "queries.csv", index=False)

    stats, nx_graphs = [], {}
    for name, edges in graphs.items():
        # edge list: the readable form - every edge and WHY it exists
        pd.DataFrame([{
            "src": i, "dst": j,
            "src_title": titles[i], "dst_title": titles[j],
            "edge_text": text,
        } for i, j, text in edges]).to_csv(out / "graphs" / f"{name}_edges.csv", index=False)

        # adjacency matrix: the same graph as a doc_id x doc_id 0/1 grid.
        # Identical node order in all four files, so subtracting two of them
        # shows exactly which edges one strategy adds over another.
        matrix = torch.zeros(n, n, dtype=torch.int8)
        for i, j, _ in edges:
            matrix[i, j] = matrix[j, i] = 1
        pd.DataFrame(matrix.numpy(), index=doc_ids, columns=doc_ids).to_csv(
            out / "graphs" / f"{name}_adjacency.csv", index_label="doc_id")

        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, j) for i, j, _ in edges)
        nx_graphs[name] = g

        # reuse the project's own stats function; it wants a PyG Data with
        # both edge directions present, which is how build() would store them
        src, dst = [i for i, _, _ in edges], [j for _, j, _ in edges]
        data = Data(edge_index=torch.tensor([src + dst, dst + src], dtype=torch.long),
                    num_nodes=n)
        stats.append(EntityGraphBuilder().compute_stats(data, queries, name=name))

    pd.DataFrame(stats).to_csv(out / "graph_stats.csv", index=False)

    # The question the project is actually asking: can this graph put a query's
    # gold documents within reach of each other?
    #   direct_edges     gold pairs joined by one edge - what a 1-hop retriever
    #                    can follow
    #   pairs_reachable  gold pairs connected by any path; below n_gold_pairs
    #                    means the evidence sits in separate components and no
    #                    amount of graph walking will ever join it
    #   max_hops         longest shortest-path between gold pairs, i.e. how far
    #                    a retriever must walk (blank if not all are reachable)
    rows = []
    for q in queries:
        pairs = list(itertools.combinations(q["gold_doc_ids"], 2))
        for name, g in nx_graphs.items():
            hops = [nx.shortest_path_length(g, a, b) if nx.has_path(g, a, b) else None
                    for a, b in pairs]
            reachable = [h for h in hops if h is not None]
            rows.append({
                "query_id": q["query_id"], "graph": name,
                "n_gold": len(q["gold_doc_ids"]), "n_gold_pairs": len(pairs),
                "direct_edges": sum(1 for h in reachable if h == 1),
                "pairs_reachable": len(reachable),
                "max_hops": max(reachable) if len(reachable) == len(pairs) else "",
            })
    connectivity = pd.DataFrame(rows)
    connectivity.to_csv(out / "gold_connectivity.csv", index=False)

    print(f"\n{'graph':<10}{'edges':>8}{'avg deg':>9}{'components':>12}"
          f"{'gold pairs w/ direct edge':>27}{'reachable at all':>19}")
    for s in stats:
        sub = connectivity[connectivity["graph"] == s["name"]]
        total = sub["n_gold_pairs"].sum()
        print(f"{s['name']:<10}{s['n_edges']:>8}{s['avg_degree']:>9.2f}{s['n_components']:>12}"
              f"{f'{sub.direct_edges.sum()}/{total}':>27}{f'{sub.pairs_reachable.sum()}/{total}':>19}")
    print("\nwrote documents.csv, queries.csv, graph_stats.csv, gold_connectivity.csv,")
    print(f"and per graph an _edges.csv and an _adjacency.csv, to {out}")


def test_build_graphs():
    """Structural check on a small slice - the real runs happen via __main__.

    Asserts wiring, not quality: all four variants share the node set, edges
    are in range and undirected, and combined really is the union.
    """
    graphs = build_graphs(PYTEST_CONFIG, n_docs=PYTEST_DOCS, n_queries=PYTEST_QUERIES)

    assert set(graphs) == {"entity", "metadata", "semantic", "combined"}
    pairs = {}
    for name, edges in graphs.items():
        assert edges, f"{name}: no edges at all"
        for i, j, text in edges:
            assert 0 <= i < PYTEST_DOCS and 0 <= j < PYTEST_DOCS, f"{name}: node out of range"
            assert i < j, f"{name}: get_edges must return each undirected pair once, as i<j"
            assert text, f"{name}: edge ({i},{j}) has no description"
        pairs[name] = {(i, j) for i, j, _ in edges}
        assert len(pairs[name]) == len(edges), f"{name}: duplicate edges"

    # combined is the union of the other three - exactly, not approximately
    union = pairs["entity"] | pairs["metadata"] | pairs["semantic"]
    assert pairs["combined"] == union, "combined is not the union of the three strategies"

    out = OUT_ROOT / f"{PYTEST_CONFIG.stem}_{PYTEST_DOCS}docs"
    for name in graphs:
        adjacency = pd.read_csv(out / "graphs" / f"{name}_adjacency.csv", index_col="doc_id")
        assert adjacency.shape == (PYTEST_DOCS, PYTEST_DOCS)
        values = adjacency.values
        assert (values == values.T).all(), f"{name}: adjacency matrix is not symmetric"
        assert (values.diagonal() == 0).all(), f"{name}: adjacency has self-loops"
        assert values.sum() == 2 * len(graphs[name]), f"{name}: matrix and edge list disagree"

    for f in ("documents.csv", "queries.csv", "graph_stats.csv", "gold_connectivity.csv"):
        assert (out / f).exists(), f"{f} was not written"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "config" / "base.yaml",
                   help="YAML config (default: config/base.yaml, the real models)")
    p.add_argument("--docs", type=int, default=None,
                   help="use only this many documents (default: the whole corpus)")
    p.add_argument("--queries", type=int, default=None,
                   help="use only this many queries (default: all of them)")
    p.add_argument("--seed", type=int, default=SEED, help="which random subset to take")
    args = p.parse_args()
    build_graphs(args.config, n_docs=args.docs, n_queries=args.queries, seed=args.seed)
