"""Shared helpers for 03_* retrieval scripts."""
import json
import logging
import os
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

_K_BASELINE_CACHE = PROCESSED / "k_baseline.json"

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
    """PCST on entity graph for 50 queries; return rounded average output size.

    Result is cached to data/processed/k_baseline.json so multiple no_pcst
    scripts don't each repeat the 50-query entity-PCST calibration run.
    """
    if _K_BASELINE_CACHE.exists():
        k = json.loads(_K_BASELINE_CACHE.read_text())["k_baseline"]
        logger.info("Loaded cached k_baseline=%d from %s", k, _K_BASELINE_CACHE.name)
        return k

    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    pcst_cfg = cfg["pcst"]
    data, tn, te = load_graph("entity")
    n_sample = min(50, len(queries))
    sizes = []
    for i in range(n_sample):
        retrieved = retrieve_with_pcst(
            query_embs[i], data, tn, te,
            topk=pcst_cfg["topk"],
            topk_e=pcst_cfg["topk_e"],
            cost_e=pcst_cfg["cost_e"],
        )
        sizes.append(len(retrieved))
    k = max(1, round(sum(sizes) / len(sizes)))

    _K_BASELINE_CACHE.write_text(json.dumps({"k_baseline": k}))
    logger.info("k_baseline=%d written to %s", k, _K_BASELINE_CACHE.name)
    return k


def compute_seed_k(graph_data, k_baseline: int) -> int:
    """Estimate seed K from median node degree.

    After 1-hop expansion the expected final size is seed_k * (1 + median_deg).
    Solving for seed_k that hits k_baseline is O(E) — no sampling needed.
    """
    n = graph_data.num_nodes or 0
    if n == 0 or graph_data.edge_index.numel() == 0:
        return k_baseline
    degrees = torch.bincount(graph_data.edge_index[0], minlength=n).float()
    median_deg = float(degrees.median().item())
    return max(1, round(k_baseline / (1.0 + median_deg)))


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


# ---------------------------------------------------------------------------
# PCST parallel helpers
# ---------------------------------------------------------------------------
# Module-level globals set in each worker process by _pcst_init.
_worker_graph_data = None
_worker_tn = None
_worker_te = None
_worker_pcst_cfg = None


def _pcst_init(root_str: str, graph_data, tn, te, pcst_cfg: dict) -> None:
    """Worker initializer: receives shared graph state once per process."""
    global _worker_graph_data, _worker_tn, _worker_te, _worker_pcst_cfg
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _worker_graph_data = graph_data
    _worker_tn = tn
    _worker_te = te
    _worker_pcst_cfg = pcst_cfg


def _pcst_call(q_emb: torch.Tensor) -> list[int]:
    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    return retrieve_with_pcst(
        q_emb, _worker_graph_data, _worker_tn, _worker_te,
        topk=_worker_pcst_cfg["topk"],
        topk_e=_worker_pcst_cfg["topk_e"],
        cost_e=_worker_pcst_cfg["cost_e"],
    )


def run_pcst_parallel(
    query_embs_list: list,
    graph_data,
    tn,
    te,
    pcst_cfg: dict,
    desc: str = "pcst",
) -> list[list[int]]:
    """Run PCST for all queries in parallel across CPU cores.

    Uses spawn context so that CUDA/PyTorch state in the parent process does
    not corrupt forked workers. Graph data is sent once per worker at init.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from tqdm import tqdm

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    logger.info("Running PCST on %d workers for %d queries", n_workers, len(query_embs_list))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_pcst_init,
        initargs=(str(ROOT), graph_data, tn, te, pcst_cfg),
    ) as pool:
        results = list(tqdm(pool.map(_pcst_call, query_embs_list), total=len(query_embs_list), desc=desc))
    return results
