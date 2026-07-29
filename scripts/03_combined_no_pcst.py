"""Phase 3: Combined graph retrieval without PCST (1-hop neighborhood expansion)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _retrieval_common import (
    PROCESSED, RETRIEVAL, logger,
    load_graph, load_embeddings, calibrate_k, compute_seed_k,
    make_result, spot_check, check_degenerate,
)

GRAPH_NAME = "combined"


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.dense_retrieval import precompute_similarities, retrieve_graph_neighborhood, build_adj

    out_path = RETRIEVAL / f"{GRAPH_NAME}_no_pcst.jsonl"
    if out_path.exists():
        logger.info("Output %s already exists, skipping.", out_path.name)
        return

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    doc_embs, query_embs = load_embeddings()

    k_baseline = calibrate_k(queries, query_embs)
    logger.info("Baseline K = %d", k_baseline)

    data, tn, te = load_graph(GRAPH_NAME)
    seed_k = compute_seed_k(data, k_baseline)
    logger.info("Seed K for %s = %d (target post-expansion ~%d)", GRAPH_NAME, seed_k, k_baseline)

    logger.info("Precomputing similarity matrix and adjacency list...")
    all_sims = precompute_similarities(query_embs, doc_embs)
    adj = build_adj(data)

    results = []
    for i, q in enumerate(queries):
        retrieved = retrieve_graph_neighborhood(all_sims[i], data, k=seed_k, expand_hops=1, adj=adj)
        results.append(make_result(q, retrieved))
        if i < 3:
            spot_check(q, retrieved)

    save_jsonl(results, out_path)
    check_degenerate(results, GRAPH_NAME, "no_pcst")
    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
