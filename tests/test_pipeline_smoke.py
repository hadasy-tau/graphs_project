"""End-to-end smoke test: runs the whole pipeline on 40 docs and 8 queries.

Everything happens inside one function, top to bottom, in the same order as
scripts/01 -> 02 -> 03 -> 04. Put a breakpoint anywhere and every variable
computed so far is sitting right there in the local scope.

    python tests/test_pipeline_smoke.py       # plain run (or F5 in VS Code)
    python -m pytest tests/test_pipeline_smoke.py -s

Results are written to data/smoke_run/. Document embeddings are cached there,
so only the first run pays for them; delete the folder to start clean.

Two things are scaled down so this finishes in under a minute on a laptop:
config/test_small.yaml uses smaller models (it explains each choice), and if
pcst-fast is not installed, tests/pcst_fallback.py substitutes an approximate
solver. Neither is suitable for real results - see tests/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))              # so `import src...` works
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` works

import pandas as pd
import torch
import yaml

import pcst_fallback
from common import setup_logging
from src.data.embedder import embed_documents, embed_queries, embed_texts, load_or_compute
from src.data.loader import load_jsonl, save_jsonl
from src.data.preprocessor import clean_text, extract_entities
from src.evaluation.evaluator import evaluate_all
from src.graph.combined_graph import CombinedGraphBuilder
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.metadata_graph import MetadataGraphBuilder
from src.graph.semantic_graph import SemanticGraphBuilder
from src.retrieval.dense_retrieval import retrieve_dense, retrieve_graph_neighborhood
from src.retrieval.pcst_retrieval import retrieve_with_pcst

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUT = ROOT / "data" / "smoke_run"
GRAPH_NAMES = ["entity", "metadata", "semantic", "combined"]


def test_pipeline_smoke():
    """Run every stage and check that each one produced something valid.

    The assertions are about structure - shapes line up, ids are in range,
    metrics are in [0, 1]. They deliberately say nothing about which graph
    retrieves better: 40 documents cannot answer that question.
    """
    setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "config" / "test_small.yaml"))

    pcst_impl = pcst_fallback.install_if_missing()
    print(f"\nPCST solver: {pcst_impl}")

    # =====================================================================
    # 1. LOAD  (scripts/01)
    # =====================================================================
    print("\n=== 1. load ===")
    corpus = load_jsonl(FIXTURES / "mini_corpus.jsonl")
    queries = load_jsonl(FIXTURES / "mini_queries.jsonl")
    n_docs = len(corpus)
    print(f"{n_docs} docs, {len(queries)} queries")
    print(f"doc 0: {corpus[0]['title']}")
    print(f"       source={corpus[0]['source']!r} category={corpus[0]['category']!r}")
    for q in queries:
        print(f"  q{q['query_id']} [{q['question_type']:<16}] gold={q['gold_doc_ids']}")

    # doc_id is used as the node index everywhere, so it must be 0..N-1
    assert [d["doc_id"] for d in corpus] == list(range(n_docs))
    for q in queries:
        assert q["gold_doc_ids"], f"q{q['query_id']} has no gold evidence"
        assert all(0 <= g < n_docs for g in q["gold_doc_ids"])

    # =====================================================================
    # 2. PREPROCESS: clean text, then spaCy NER  (scripts/01)
    # =====================================================================
    print("\n=== 2. preprocess ===")
    for doc in corpus:
        doc["body"] = clean_text(doc["body"], max_tokens=cfg["embedding"]["max_tokens"])

    entities = extract_entities(
        corpus,
        model_name=cfg["ner"]["model"],
        batch_size=cfg["ner"]["batch_size"],
        keep_types=frozenset(cfg["ner"]["keep_types"]),
        stop_entity_threshold=cfg["ner"]["stop_entity_threshold"],
        canonicalize_threshold=cfg["ner"]["canonicalize_threshold"],
    )
    print(f"doc 0 entities: {sorted(entities[0])[:8]}")

    assert set(entities) == set(range(n_docs))
    n_empty = sum(1 for e in entities.values() if not e)
    assert n_empty <= 0.1 * n_docs, f"{n_empty}/{n_docs} docs got no entities - NER looks broken"

    # =====================================================================
    # 3. EMBED documents and queries  (scripts/01)
    # =====================================================================
    print("\n=== 3. embed ===")
    dim = cfg["embedding"]["dim"]
    doc_embs = load_or_compute(
        OUT / "embeddings.pt", embed_documents, corpus,
        model_name=cfg["embedding"]["model"], batch_size=cfg["embedding"]["batch_size"],
    )
    query_embs = load_or_compute(
        OUT / "query_embeddings.pt", embed_queries, queries,
        model_name=cfg["embedding"]["model"], batch_size=cfg["embedding"]["batch_size"],
    )
    print(f"doc_embs={tuple(doc_embs.shape)} query_embs={tuple(query_embs.shape)}")

    # sanity peek: are q0's nearest documents anywhere near its gold documents?
    top = torch.topk(query_embs[0] @ doc_embs.T, 3)
    print(f"q0 nearest: {top.indices.tolist()} (sims {[round(s, 3) for s in top.values.tolist()]})")
    print(f"q0 gold   : {queries[0]['gold_doc_ids']}")

    assert doc_embs.shape == (n_docs, dim)
    assert query_embs.shape == (len(queries), dim)
    assert torch.isfinite(doc_embs).all()
    assert torch.allclose(doc_embs.norm(dim=1), torch.ones(n_docs), atol=1e-3)

    # =====================================================================
    # 4. BUILD the four graph variants  (scripts/02)
    # =====================================================================
    print("\n=== 4. build graphs ===")

    def embed_fn(texts):
        return embed_texts(texts, model_name=cfg["embedding"]["model"],
                           batch_size=cfg["embedding"]["batch_size"])

    entity_builder = EntityGraphBuilder(cfg["entity_graph"]["min_shared_entities"])
    metadata_builder = MetadataGraphBuilder()

    # The entity graph is built first because its density sets the semantic
    # graph's k - otherwise the two would differ in edge count as well as in
    # edge meaning, and you could not tell which caused a difference.
    entity_graph = entity_builder.build(corpus, doc_embs, entities, edge_embedder=embed_fn)
    avg_degree = entity_graph[0].edge_index.shape[1] / n_docs
    k = cfg["semantic_graph"]["mutual_knn_k"] or max(1, round(avg_degree))
    print(f"entity avg degree {avg_degree:.2f} -> semantic mutual_knn_k = {k}")

    semantic_builder = SemanticGraphBuilder(cfg["semantic_graph"]["threshold"], mutual_knn_k=k)
    combined_builder = CombinedGraphBuilder(entity_builder, metadata_builder, semantic_builder)

    graphs = {
        "entity": entity_graph,
        "metadata": metadata_builder.build(corpus, doc_embs, entities, edge_embedder=embed_fn),
        "semantic": semantic_builder.build(corpus, doc_embs, entities, edge_embedder=embed_fn),
        "combined": combined_builder.build(corpus, doc_embs, entities, edge_embedder=embed_fn),
    }

    for name, (data, nodes_df, edges_df) in graphs.items():
        stats = entity_builder.compute_stats(data, queries, name=name)
        print(f"[{name:<9}] edges={stats['n_edges']:<4} components={stats['n_components']:<3} "
              f"oracle_conn={stats['oracle_connectivity']}")
        print(f"            e.g. {edges_df['edge_attr'][0][:72]!r}")

        assert data.num_nodes == n_docs, f"{name}: all variants must share the node set"
        assert data.edge_index.shape[1] > 0, f"{name}: no edges"
        assert data.edge_index.shape[1] % 2 == 0, f"{name}: edges must be symmetric"
        assert data.edge_attr.shape == (data.edge_index.shape[1], dim)
        assert len(edges_df) == data.edge_index.shape[1]
        assert len(nodes_df) == n_docs

    # combined is the union of the other three, so it cannot have fewer edges
    n_edges = {name: g[0].edge_index.shape[1] for name, g in graphs.items()}
    assert n_edges["combined"] >= max(n_edges[n] for n in ("entity", "metadata", "semantic"))

    # oracle_connectivity = share of gold document PAIRS joined by a direct
    # edge. It is the ceiling on what a single graph hop can do for a
    # multi-hop question, so it is the most informative number printed above.

    # =====================================================================
    # 5. SAVE and RELOAD - the scripts/02 -> scripts/03 handoff
    # =====================================================================
    print("\n=== 5. save + reload ===")
    graph_dir = OUT / "graphs"
    graph_dir.mkdir(exist_ok=True)
    for name, (data, nodes_df, edges_df) in graphs.items():
        torch.save(data, graph_dir / f"{name}_graph.pt")
        nodes_df.to_csv(graph_dir / f"{name}_textual_nodes.csv", index=False)
        edges_df.to_csv(graph_dir / f"{name}_textual_edges.csv", index=False)

        graphs[name] = (
            torch.load(graph_dir / f"{name}_graph.pt", weights_only=False),
            pd.read_csv(graph_dir / f"{name}_textual_nodes.csv"),
            pd.read_csv(graph_dir / f"{name}_textual_edges.csv"),
        )
        # PCST indexes the DataFrames by node/edge position, so a CSV round
        # trip that drops or reorders a row would silently corrupt retrieval
        data, nodes_df, edges_df = graphs[name]
        assert len(nodes_df) == data.num_nodes, f"{name}: node rows lost in round trip"
        assert len(edges_df) == data.edge_index.shape[1], f"{name}: edge rows lost"
    print("all four graphs round-tripped with rows still aligned")

    # =====================================================================
    # 6. RETRIEVE under all 9 conditions  (scripts/03)
    # =====================================================================
    print("\n=== 6. retrieval ===")
    pcst_cfg = cfg["pcst"]
    retrieval_dir = OUT / "retrieval"
    retrieval_dir.mkdir(exist_ok=True)

    def run_pcst(q_emb, graph_name):
        data, nodes_df, edges_df = graphs[graph_name]
        return retrieve_with_pcst(q_emb, data, nodes_df, edges_df, topk=pcst_cfg["topk"],
                                  topk_e=pcst_cfg["topk_e"], cost_e=pcst_cfg["cost_e"])

    # The non-PCST conditions need a K. Match it to how many documents PCST
    # returns on average, or a condition could win on recall just by returning
    # more documents.
    sizes = [len(run_pcst(query_embs[i], "entity")) for i in range(len(queries))]
    k_baseline = cfg["dense_baseline"]["k"] or max(1, round(sum(sizes) / len(sizes)))
    print(f"entity-PCST returned {sizes} docs -> baseline K = {k_baseline}")

    conditions = ([("dense", None, "no_pcst")]
                  + [(n, n, "pcst") for n in GRAPH_NAMES]
                  + [(n, n, "no_pcst") for n in GRAPH_NAMES])

    for label, graph_name, mode in conditions:
        results = []
        for i, q in enumerate(queries):
            if graph_name is None:
                retrieved = retrieve_dense(query_embs[i], doc_embs, k=k_baseline)
            elif mode == "pcst":
                retrieved = run_pcst(query_embs[i], graph_name)
            else:
                retrieved = retrieve_graph_neighborhood(
                    query_embs[i], doc_embs, graphs[graph_name][0], k=k_baseline, expand_hops=1)
            results.append({
                "query_id": q["query_id"], "query": q["query"],
                "question_type": q["question_type"],
                "retrieved_doc_ids": sorted(retrieved),
                "gold_doc_ids": q["gold_doc_ids"],
            })
        save_jsonl(results, retrieval_dir / f"{label}_{mode}.jsonl")

        for r in results:
            assert r["retrieved_doc_ids"], f"{label}_{mode} q{r['query_id']}: retrieved nothing"
            assert all(0 <= d < n_docs for d in r["retrieved_doc_ids"])
            assert len(set(r["retrieved_doc_ids"])) == len(r["retrieved_doc_ids"])

        if label == "combined" and mode == "pcst":
            print("per-query detail for combined_pcst:")
            for r in results:
                gold, got = set(r["gold_doc_ids"]), set(r["retrieved_doc_ids"])
                print(f"  q{r['query_id']} gold={sorted(gold)} got={sorted(got)} "
                      f"missed={sorted(gold - got)}")

    assert len(list(retrieval_dir.glob("*.jsonl"))) == 9

    # =====================================================================
    # 7. EVALUATE  (scripts/04)
    # =====================================================================
    print("\n=== 7. evaluate ===")
    evaluate_all(retrieval_dir, OUT / "metrics")
    summary = pd.read_csv(OUT / "metrics" / "summary_table.csv")

    print(f"{'condition':<20}{'recall':>8}{'hit_rate':>10}{'prec':>8}{'f1':>8}{'size':>7}")
    for _, r in summary.iterrows():
        print(f"{r['condition']:<20}{r['evidence_recall']:>8.3f}{r['full_hit_rate']:>10.3f}"
              f"{r['precision']:>8.3f}{r['f1']:>8.3f}{r['avg_retrieved_size']:>7.1f}")

    assert len(summary) == 9
    for _, r in summary.iterrows():
        for col in ("evidence_recall", "full_hit_rate", "precision", "f1"):
            assert 0.0 <= r[col] <= 1.0, f"{r['condition']}.{col} = {r[col]}"
        assert r["n_queries"] == len(queries)
        # you cannot fully answer more queries than you partially answer
        assert r["full_hit_rate"] <= r["evidence_recall"] + 1e-9, r["condition"]

    assert (OUT / "metrics" / "by_qtype_table.csv").exists()
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    test_pipeline_smoke()
