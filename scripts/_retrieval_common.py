"""Shared helpers for 03_* retrieval scripts."""
import hashlib
import json
import logging
import os
import sys

# Resolves the active config and every output path from the environment, and puts
# the repo root on sys.path. Import before anything from src/.
# RETRIEVAL is re-exported: the nine 03_* scripts import it from here.
from experiment import GRAPHS, PROCESSED, RETRIEVAL, ROOT, cfg, ensure_dirs  # noqa: F401

import torch

logger = logging.getLogger(__name__)

# Every 03_* script imports this module, so one call covers them all.
ensure_dirs()

# Both are derived from the graphs (k_baseline is the average entity-PCST output
# size; seed_k is calibrated against a graph's expansion ratio), so they live with
# the graphs. In data/processed they would be shared by experiments whose graphs
# differ, and the cache keys below do not mention the graph config.
_K_BASELINE_CACHE = GRAPHS / "k_baseline.json"
_SEED_K_CACHE = GRAPHS / "seed_k.json"


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


def _pcst_config_hash() -> str:
    """Short hash of the PCST config so the cache busts when params change."""
    pcst_cfg = cfg.get("pcst", {})
    blob = json.dumps(pcst_cfg, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def calibrate_k(queries, query_embs) -> int:
    """PCST on entity graph for 50 queries; return rounded average output size.

    Result is cached to GRAPHS/k_baseline.json keyed by a hash of the PCST
    config, so the cache auto-invalidates when topk/cost_e/topk_e change.
    """
    cfg_hash = _pcst_config_hash()
    cache_key = f"k_baseline_{cfg_hash}"

    if _K_BASELINE_CACHE.exists():
        cache = json.loads(_K_BASELINE_CACHE.read_text())
        if cache_key in cache:
            k = cache[cache_key]
            logger.info("Loaded cached k_baseline=%d (cfg_hash=%s)", k, cfg_hash)
            return k
        logger.info("PCST config changed (cfg_hash=%s); re-calibrating k_baseline", cfg_hash)
    else:
        cache = {}

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

    cache[cache_key] = k
    _K_BASELINE_CACHE.write_text(json.dumps(cache))
    logger.info("k_baseline=%d (cfg_hash=%s) written to %s", k, cfg_hash, _K_BASELINE_CACHE.name)
    return k


def calibrate_seed_k(
    all_sims: torch.Tensor,
    graph_data,
    adj: dict,
    k_baseline: int,
    graph_name: str,
    n_sample: int = 50,
) -> int:
    """Empirically find seed_k so the mean post-expansion size ≈ k_baseline.

    Runs the actual graph neighborhood expansion on n_sample queries using
    seed_k = k_baseline as an upper bound, measures the mean expansion ratio,
    then solves: seed_k = round(k_baseline / ratio).

    Result is cached per (graph_name, k_baseline) so it only runs once.
    """
    from src.retrieval.dense_retrieval import retrieve_graph_neighborhood

    cache_key = f"{graph_name}_{k_baseline}"
    cache: dict = json.loads(_SEED_K_CACHE.read_text()) if _SEED_K_CACHE.exists() else {}
    if cache_key in cache:
        k = cache[cache_key]
        logger.info("Loaded cached seed_k[%s]=%d", graph_name, k)
        return k

    if graph_data.edge_index.numel() == 0:
        logger.info("seed_k[%s]=%d (no edges — no expansion)", graph_name, k_baseline)
        cache[cache_key] = k_baseline
        _SEED_K_CACHE.write_text(json.dumps(cache))
        return k_baseline

    n_sample = min(n_sample, all_sims.shape[0])
    indices = torch.linspace(0, all_sims.shape[0] - 1, n_sample).long().tolist()
    sizes = [
        len(retrieve_graph_neighborhood(
            all_sims[i], graph_data, k=k_baseline, expand_hops=1, adj=adj
        ))
        for i in indices
    ]
    mean_size = sum(sizes) / len(sizes)
    ratio = mean_size / k_baseline
    seed_k = max(1, round(k_baseline / ratio))

    logger.info(
        "seed_k[%s]: mean_expansion=%.1f ratio=%.2f -> seed_k=%d",
        graph_name, mean_size, ratio, seed_k,
    )
    cache[cache_key] = seed_k
    _SEED_K_CACHE.write_text(json.dumps(cache))
    return seed_k


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
