"""Phase 3: Run all retrieval conditions and save results to JSONL files."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Resolves the active config and every output path from the environment, and puts
# the repo root on sys.path. Import before anything from src/.
from experiment import GRAPHS, PROCESSED, RETRIEVAL, cfg

import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    from src.data.loader import load_jsonl, save_jsonl
    from src.retrieval.dense_retrieval import (
        precompute_similarities, build_adj,
        retrieve_dense, retrieve_graph_neighborhood,
    )
    from _retrieval_common import (
        calibrate_k, calibrate_seed_k,
        make_result, spot_check, check_degenerate, run_pcst_parallel,
    )

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    doc_embs = torch.load(PROCESSED / "embeddings.pt", weights_only=True)
    query_embs = torch.load(PROCESSED / "query_embeddings.pt", weights_only=True)

    # --- Load all graphs ---
    base_graph_names = ["entity", "metadata", "semantic", "combined"]
    graph_names = list(base_graph_names)
    if cfg.get("null_baselines", False):
        graph_names += [f"{n}_{s}" for n in base_graph_names
                        for s in ("random_structure", "shuffled_nodes")]
    logger.info("Graphs for this arm (%d): %s", len(graph_names), graph_names)
    graphs = {}
    for name in graph_names:
        data = torch.load(GRAPHS / f"{name}_graph.pt", weights_only=False)
        tn = pd.read_csv(GRAPHS / f"{name}_textual_nodes.csv")
        te = pd.read_csv(GRAPHS / f"{name}_textual_edges.csv")
        graphs[name] = (data, tn, te)

    # --- Calibrate K (cached after first run) ---
    k_baseline = calibrate_k(queries, query_embs)
    logger.info("Baseline K = %d", k_baseline)

    # --- Precompute similarities once for all no-PCST conditions ---
    logger.info("Precomputing similarity matrix (%d x %d)...", query_embs.shape[0], doc_embs.shape[0])
    all_sims = precompute_similarities(query_embs, doc_embs)

    # --- Build adjacency lists, then empirically calibrate seed_k per graph ---
    adj_by_graph = {}
    for name in graph_names:
        data, _, _ = graphs[name]
        adj_by_graph[name] = build_adj(data)

    seed_k_by_graph = {}
    for name in graph_names:
        data, _, _ = graphs[name]
        seed_k_by_graph[name] = calibrate_seed_k(
            all_sims, data, adj_by_graph[name], k_baseline, graph_name=name,
        )
        logger.info(
            "Seed K for %s = %d (target post-expansion ~%d)",
            name, seed_k_by_graph[name], k_baseline,
        )

    # Per-graph PCST config: combined graph gets a lower edge cost because its
    # multi-type edges carry more signal per traversal than single-type edges.
    pcst_cfg_by_graph = {name: dict(cfg["pcst"]) for name in graph_names}
    if "combined_cost_e" in cfg["pcst"]:
        # applies to combined AND its nulls (combined_random_structure, ...)
        for _n in graph_names:
            if _n == "combined" or _n.startswith("combined_"):
                pcst_cfg_by_graph[_n]["cost_e"] = cfg["pcst"]["combined_cost_e"]
    for name, pcfg in pcst_cfg_by_graph.items():
        logger.info("PCST cfg for %s: topk=%d cost_e=%.2f", name, pcfg["topk"], pcfg["cost_e"])

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

        if mode == "pcst":
            data, tn, te = graphs[graph_name]
            all_retrieved = run_pcst_parallel(
                [query_embs[i] for i in range(len(queries))],
                data, tn, te, pcst_cfg_by_graph[graph_name],
                desc=f"{label}/pcst",
            )
            results = []
            for i, (q, retrieved) in enumerate(zip(queries, all_retrieved)):
                results.append(make_result(q, retrieved))
                if i < 3:
                    spot_check(q, retrieved)
        else:
            results = []
            for i, q in enumerate(tqdm(queries, desc=f"{label}/{mode}")):
                if graph_name is None:
                    retrieved = retrieve_dense(all_sims[i], k=k_baseline)
                else:
                    data, _, _ = graphs[graph_name]
                    retrieved = retrieve_graph_neighborhood(
                        all_sims[i], data, k=seed_k_by_graph[graph_name],
                        expand_hops=1, adj=adj_by_graph[graph_name],
                        rerank_to=k_baseline,
                    )
                results.append(make_result(q, retrieved))
                if i < 3:
                    spot_check(q, retrieved)

        save_jsonl(results, out_path)
        check_degenerate(results, label, mode)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
