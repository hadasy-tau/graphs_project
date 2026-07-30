"""Phase 3: Combined graph retrieval with PCST."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _retrieval_common import (
    PROCESSED, RETRIEVAL, cfg, logger,
    load_graph, load_embeddings, make_result, spot_check, check_degenerate,
    run_pcst_parallel,
)

GRAPH_NAME = "combined"


def main():
    from src.data.loader import load_jsonl, save_jsonl

    out_path = RETRIEVAL / f"{GRAPH_NAME}_pcst.jsonl"
    if out_path.exists():
        logger.info("Output %s already exists, skipping.", out_path.name)
        return

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    _, query_embs = load_embeddings()
    data, tn, te = load_graph(GRAPH_NAME)

    # The combined graph gets a lower edge cost: its multi-type edges carry more
    # signal per traversal. Without this, this script and 03_run_retrieval.py
    # would write different numbers to the same combined_pcst.jsonl.
    pcst_cfg = dict(cfg["pcst"])
    if "combined_cost_e" in pcst_cfg:
        pcst_cfg["cost_e"] = pcst_cfg["combined_cost_e"]
    logger.info("PCST cfg for %s: topk=%d cost_e=%.2f",
                GRAPH_NAME, pcst_cfg["topk"], pcst_cfg["cost_e"])

    all_retrieved = run_pcst_parallel(
        [query_embs[i] for i in range(len(queries))],
        data, tn, te, pcst_cfg,
        desc=f"{GRAPH_NAME}/pcst",
    )

    results = []
    for i, (q, retrieved) in enumerate(zip(queries, all_retrieved)):
        results.append(make_result(q, retrieved))
        if i < 3:
            spot_check(q, retrieved)

    save_jsonl(results, out_path)
    check_degenerate(results, GRAPH_NAME, "pcst")
    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
