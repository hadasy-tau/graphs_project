"""Phase 3: Combined graph retrieval with PCST."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from _retrieval_common import (
    PROCESSED, RETRIEVAL, cfg, logger,
    load_graph, load_embeddings, make_result, spot_check, check_degenerate,
)

GRAPH_NAME = "combined"


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.pcst_retrieval import retrieve_with_pcst

    out_path = RETRIEVAL / f"{GRAPH_NAME}_pcst.jsonl"
    if out_path.exists():
        logger.info("Output %s already exists, skipping.", out_path.name)
        return

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    _, query_embs = load_embeddings()
    pcst_cfg = cfg["pcst"]
    data, tn, te = load_graph(GRAPH_NAME)

    results = []
    for i, q in enumerate(tqdm(queries, desc=f"{GRAPH_NAME}/pcst")):
        retrieved = retrieve_with_pcst(
            query_embs[i], data, tn, te,
            topk=pcst_cfg["topk"],
            topk_e=pcst_cfg["topk_e"],
            cost_e=pcst_cfg["cost_e"],
        )
        results.append(make_result(q, retrieved))
        if i < 3:
            spot_check(q, retrieved)

    save_jsonl(results, out_path)
    check_degenerate(results, GRAPH_NAME, "pcst")
    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
