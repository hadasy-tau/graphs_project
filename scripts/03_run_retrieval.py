"""Phase 3: Run all retrieval conditions and save results to JSONL files."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = ROOT / "data" / "processed"
GRAPHS = ROOT / "data" / "graphs"
RETRIEVAL = ROOT / "results" / "retrieval"
RETRIEVAL.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "base.yaml") as f:
    cfg = yaml.safe_load(f)


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    from src.retrieval.dense_retrieval import retrieve_dense, retrieve_graph_neighborhood

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    doc_embs = torch.load(PROCESSED / "embeddings.pt", weights_only=True)
    query_embs = torch.load(PROCESSED / "query_embeddings.pt", weights_only=True)

    pcst_cfg = cfg["pcst"]

    # --- Load all graphs ---
    graph_names = ["entity", "metadata", "semantic", "combined"]
    graphs = {}
    for name in graph_names:
        data = torch.load(GRAPHS / f"{name}_graph.pt", weights_only=False)
        tn = pd.read_csv(GRAPHS / f"{name}_textual_nodes.csv")
        te = pd.read_csv(GRAPHS / f"{name}_textual_edges.csv")
        graphs[name] = (data, tn, te)

    # --- Calibrate K for dense / no-PCST baselines ---
    k_baseline = _calibrate_k(queries, query_embs, graphs["entity"], pcst_cfg)
    logger.info("Baseline K calibrated to %d (matches entity-PCST avg output size)", k_baseline)

    # --- Calibrate seed K per graph so that the POST-EXPANSION size (seeds +
    # 1-hop neighbors) matches k_baseline, instead of using k_baseline as the
    # seed count itself (which would always over-shoot after expansion). ---
    seed_k_by_graph = {}
    for name in graph_names:
        data, _, _ = graphs[name]
        seed_k_by_graph[name] = _calibrate_seed_k(query_embs, doc_embs, data, target_size=k_baseline)
        logger.info(
            "Seed K for %s calibrated to %d (target final size ~%d after 1-hop expansion)",
            name, seed_k_by_graph[name], k_baseline,
        )

    # --- Define all conditions ---
    conditions: list[tuple[str, str | None, str]] = (
        [("dense", None, "no_pcst")]
        + [(name, name, "pcst") for name in graph_names]
        + [(name, name, "no_pcst") for name in graph_names]
    )

    for label, graph_name, mode in conditions:
        out_path = RETRIEVAL / f"{label}_{mode}.jsonl"
        if out_path.exists():
            logger.info("Skipping %s (already exists)", out_path.name)
            continue

        logger.info("Running condition: %s / %s", label, mode)
        results = []

        for i, q in enumerate(tqdm(queries, desc=f"{label}/{mode}")):
            q_emb = query_embs[i]

            if mode == "no_pcst" and graph_name is None:
                # Pure dense baseline
                retrieved = retrieve_dense(q_emb, doc_embs, k=k_baseline)
            elif mode == "pcst":
                data, tn, te = graphs[graph_name]
                retrieved = retrieve_with_pcst(
                    q_emb, data, tn, te,
                    topk=pcst_cfg["topk"],
                    topk_e=pcst_cfg["topk_e"],
                    cost_e=pcst_cfg["cost_e"],
                )
            else:
                # no-PCST graph condition: top-K + 1-hop expansion
                data, _, _ = graphs[graph_name]
                retrieved = retrieve_graph_neighborhood(
                    q_emb, doc_embs, data, k=seed_k_by_graph[graph_name], expand_hops=1
                )

            results.append({
                "query_id": q["query_id"],
                "query": q["query"],
                "question_type": q["question_type"],
                "retrieved_doc_ids": retrieved,
                "gold_doc_ids": q["gold_doc_ids"],
            })

            if i < 3:
                _spot_check(q, retrieved)

        save_jsonl(results, out_path)
        _check_degenerate(results, label, mode)

    logger.info("Phase 3 complete.")


def _calibrate_k(queries, query_embs, entity_graph_tuple, pcst_cfg) -> int:
    """Run PCST on entity graph for first 50 queries; return rounded average output size."""
    from src.retrieval.pcst_retrieval import retrieve_with_pcst

    data, tn, te = entity_graph_tuple
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
    k = max(1, round(sum(sizes) / len(sizes)))
    return k


def _calibrate_seed_k(query_embs, doc_embs, graph_data, target_size: int, n_sample: int = 50) -> int:
    """Binary-search the seed K for retrieve_graph_neighborhood so that the average
    POST-EXPANSION size (seeds + 1-hop neighbors) matches target_size, rather than
    using target_size as the seed count itself (which always over-shoots after
    expansion, and by a different amount per graph depending on its density).
    """
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


def _spot_check(q: dict, retrieved: list[int]) -> None:
    gold = set(q["gold_doc_ids"])
    hits = gold & set(retrieved)
    logger.info(
        "  Q: %s... | gold=%s retrieved=%s hits=%s",
        q["query"][:60], sorted(gold), sorted(retrieved)[:5], sorted(hits),
    )


def _check_degenerate(results: list[dict], label: str, mode: str) -> None:
    singleton = sum(1 for r in results if len(r["retrieved_doc_ids"]) < 2)
    frac = singleton / len(results)
    if frac > 0.05:
        logger.warning(
            "[%s/%s] %.1f%% of queries returned <2 docs (degenerate PCST)",
            label, mode, frac * 100,
        )
    else:
        logger.info("[%s/%s] degenerate rate: %.1f%%", label, mode, frac * 100)


if __name__ == "__main__":
    main()
