"""Shared helpers for 03_* retrieval scripts."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = ROOT / "data" / "processed"
GRAPHS = ROOT / "data" / "graphs"
RETRIEVAL = ROOT / "results" / "retrieval"
RETRIEVAL.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "base.yaml") as f:
    cfg = yaml.safe_load(f)


def load_graph(name: str):
    import pandas as pd
    data = torch.load(GRAPHS / f"{name}_graph.pt", weights_only=False)
    tn = pd.read_csv(GRAPHS / f"{name}_textual_nodes.csv")
    te = pd.read_csv(GRAPHS / f"{name}_textual_edges.csv")
    return data, tn, te


def load_embeddings():
    doc_embs = torch.load(PROCESSED / "embeddings.pt", weights_only=True)
    query_embs = torch.load(PROCESSED / "query_embeddings.pt", weights_only=True)
    return doc_embs, query_embs


def calibrate_k(queries, query_embs) -> int:
    """Run PCST on entity graph for first 50 queries; return rounded average output size."""
    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    pcst_cfg = cfg["pcst"]
    data, tn, te = load_graph("entity")
    sizes = []
    n_sample = min(50, len(queries))
    for i in range(n_sample):
        retrieved = retrieve_with_pcst(
            query_embs[i], data, tn, te,
            topk=pcst_cfg["topk"],
            topk_e=pcst_cfg["topk_e"],
            cost_e=pcst_cfg["cost_e"],
        )
        sizes.append(len(retrieved))
    return max(1, round(sum(sizes) / len(sizes)))


def calibrate_seed_k(query_embs, doc_embs, graph_data, target_size: int, n_sample: int = 50) -> int:
    """Binary-search seed K so that post-expansion size (seeds + 1-hop) matches target_size."""
    from src.retrieval.dense_retrieval import retrieve_graph_neighborhood
    n_sample = min(n_sample, query_embs.shape[0])
    n_docs = doc_embs.shape[0]

    def avg_final_size(seed_k: int) -> float:
        sizes = []
        for i in range(n_sample):
            retrieved = retrieve_graph_neighborhood(
                query_embs[i], doc_embs, graph_data, k=seed_k, expand_hops=1
            )
            sizes.append(len(retrieved))
        return sum(sizes) / len(sizes)

    lo, hi = 1, n_docs
    best_k, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        size = avg_final_size(mid)
        diff = abs(size - target_size)
        if diff < best_diff:
            best_k, best_diff = mid, diff
        if size < target_size:
            lo = mid + 1
        elif size > target_size:
            hi = mid - 1
        else:
            break
    return max(1, best_k)


def make_result(q: dict, retrieved: list[int]) -> dict:
    return {
        "query_id": q["query_id"],
        "query": q["query"],
        "question_type": q["question_type"],
        "retrieved_doc_ids": retrieved,
        "gold_doc_ids": q["gold_doc_ids"],
    }


def spot_check(q: dict, retrieved: list[int]) -> None:
    gold = set(q["gold_doc_ids"])
    hits = gold & set(retrieved)
    logger.info(
        "  Q: %s... | gold=%s retrieved=%s hits=%s",
        q["query"][:60], sorted(gold), sorted(retrieved)[:5], sorted(hits),
    )


def check_degenerate(results: list[dict], label: str, mode: str) -> None:
    singleton = sum(1 for r in results if len(r["retrieved_doc_ids"]) < 2)
    frac = singleton / len(results)
    if frac > 0.05:
        logger.warning(
            "[%s/%s] %.1f%% of queries returned <2 docs (degenerate PCST)",
            label, mode, frac * 100,
        )
    else:
        logger.info("[%s/%s] degenerate rate: %.1f%%", label, mode, frac * 100)
