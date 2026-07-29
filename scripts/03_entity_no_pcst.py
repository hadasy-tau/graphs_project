"""Phase 3: Entity graph retrieval without PCST (1-hop neighborhood expansion)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from _retrieval_common import (
    PROCESSED, RETRIEVAL, logger,
    load_graph, load_embeddings, calibrate_k, calibrate_seed_k,
    make_result, spot_check, check_degenerate,
)

GRAPH_NAME = "entity"


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.dense_retrieval import retrieve_graph_neighborhood

    out_path = RETRIEVAL / f"{GRAPH_NAME}_no_pcst.jsonl"
    if out_path.exists():
        logger.info("Output %s already exists, skipping.", out_path.name)
        return

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    doc_embs, query_embs = load_embeddings()

    k_baseline = calibrate_k(queries, query_embs)
    logger.info("Baseline K calibrated to %d", k_baseline)

    data, tn, te = load_graph(GRAPH_NAME)
    seed_k = calibrate_seed_k(query_embs, doc_embs, data, target_size=k_baseline)
    logger.info("Seed K for %s calibrated to %d (target final size ~%d)", GRAPH_NAME, seed_k, k_baseline)

    results = []
    for i, q in enumerate(tqdm(queries, desc=f"{GRAPH_NAME}/no_pcst")):
        retrieved = retrieve_graph_neighborhood(
            query_embs[i], doc_embs, data, k=seed_k, expand_hops=1
        )
        results.append(make_result(q, retrieved))
        if i < 3:
            spot_check(q, retrieved)

    save_jsonl(results, out_path)
    check_degenerate(results, GRAPH_NAME, "no_pcst")
    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
