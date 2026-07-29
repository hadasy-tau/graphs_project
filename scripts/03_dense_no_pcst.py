"""Phase 3: Dense retrieval baseline (no graph, no PCST)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _retrieval_common import (
    PROCESSED, RETRIEVAL, logger,
    load_embeddings, calibrate_k, make_result, spot_check, check_degenerate,
)


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.dense_retrieval import precompute_similarities, retrieve_dense

    out_path = RETRIEVAL / "dense_no_pcst.jsonl"
    if out_path.exists():
        logger.info("Output %s already exists, skipping.", out_path.name)
        return

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    doc_embs, query_embs = load_embeddings()

    k_baseline = calibrate_k(queries, query_embs)
    logger.info("Baseline K = %d", k_baseline)

    logger.info("Precomputing similarity matrix (%d queries x %d docs)...", query_embs.shape[0], doc_embs.shape[0])
    all_sims = precompute_similarities(query_embs, doc_embs)

    results = []
    for i, q in enumerate(queries):
        retrieved = retrieve_dense(all_sims[i], k=k_baseline)
        results.append(make_result(q, retrieved))
        if i < 3:
            spot_check(q, retrieved)

    save_jsonl(results, out_path)
    check_degenerate(results, "dense", "no_pcst")
    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
