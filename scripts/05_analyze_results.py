"""Phase 5: Generate analysis tables, sensitivity sweeps, and error analysis."""
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from _retrieval_common import RETRIEVAL, METRICS

PROCESSED = ROOT / "data" / "processed"
GRAPHS = ROOT / "data" / "graphs"

with open(ROOT / "config" / "base.yaml") as f:
    cfg = yaml.safe_load(f)


def main():
    from src.data.loader import load_jsonl
    from src.evaluation.metrics import aggregate

    queries = load_jsonl(PROCESSED / "queries.jsonl")
    embeddings = torch.load(PROCESSED / "embeddings.pt", weights_only=True)

    # --- PCST ablation table ---
    graph_names = ["entity", "metadata", "semantic", "combined"]
    ablation_rows = []
    for name in graph_names:
        pcst_path = RETRIEVAL / f"{name}_pcst.jsonl"
        nopcst_path = RETRIEVAL / f"{name}_no_pcst.jsonl"
        if pcst_path.exists() and nopcst_path.exists():
            pcst_m = aggregate(load_jsonl(pcst_path))
            nopcst_m = aggregate(load_jsonl(nopcst_path))
            ablation_rows.append({
                "graph": name,
                "pcst_recall": pcst_m["evidence_recall"],
                "nopcst_recall": nopcst_m["evidence_recall"],
                "pcst_f1": pcst_m["f1"],
                "nopcst_f1": nopcst_m["f1"],
                "pcst_size": pcst_m["avg_retrieved_size"],
                "nopcst_size": nopcst_m["avg_retrieved_size"],
            })

    _write_csv(ablation_rows, METRICS / "pcst_ablation.csv")
    logger.info("PCST ablation table saved")

    # --- Semantic mutual k-NN sensitivity ---
    knn_rows = _sweep_semantic_knn(queries, embeddings)
    _write_csv(knn_rows, METRICS / "knn_sensitivity.csv")

    # --- Entity min-overlap sensitivity ---
    overlap_rows = _sweep_entity_overlap(queries, embeddings)
    _write_csv(overlap_rows, METRICS / "entity_overlap_sensitivity.csv")

    # --- Error analysis: queries where combined wins but entity fails ---
    _error_analysis(queries)

    logger.info("Phase 5 complete.")


def _sweep_semantic_knn(queries, embeddings):
    from src.graph.semantic_graph import SemanticGraphBuilder
    from src.data.loader import load_jsonl
    from src.data.embedder import embed_texts
    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    from src.evaluation.metrics import aggregate

    with open(PROCESSED / "entities.json") as f:
        entities = {int(k): set(v) for k, v in json.load(f).items()}
    corpus = load_jsonl(PROCESSED / "corpus.jsonl")
    query_embs = torch.load(PROCESSED / "query_embeddings.pt", weights_only=True)

    pcst_cfg = cfg["pcst"]
    embed_fn = lambda texts: embed_texts(
        texts, model_name=cfg["embedding"]["model"], batch_size=cfg["embedding"]["batch_size"]
    )

    rows = []
    for k in [3, 5, 8, 10, 15, 20]:
        builder = SemanticGraphBuilder(mutual_knn_k=k)
        data, tn, te = builder.build(corpus, embeddings, entities, edge_embedder=embed_fn)

        results = []
        for i, q in enumerate(queries):
            retrieved = retrieve_with_pcst(
                query_embs[i], data, tn, te,
                topk=pcst_cfg["topk"], topk_e=pcst_cfg["topk_e"], cost_e=pcst_cfg["cost_e"],
            )
            results.append({
                "retrieved_doc_ids": retrieved,
                "gold_doc_ids": q["gold_doc_ids"],
                "question_type": q["question_type"],
            })

        m = aggregate(results)
        rows.append({"mutual_knn_k": k, "n_edges": data.edge_index.shape[1] // 2, **m})
        logger.info("mutual_knn_k=%d -> F1=%.4f recall=%.4f edges=%d",
                    k, m["f1"], m["evidence_recall"], data.edge_index.shape[1] // 2)

    return rows


def _sweep_entity_overlap(queries, embeddings):
    from src.graph.entity_graph import EntityGraphBuilder
    from src.data.loader import load_jsonl
    from src.data.embedder import embed_texts
    from src.retrieval.pcst_retrieval import retrieve_with_pcst
    from src.evaluation.metrics import aggregate

    with open(PROCESSED / "entities.json") as f:
        entities = {int(k): set(v) for k, v in json.load(f).items()}
    corpus = load_jsonl(PROCESSED / "corpus.jsonl")
    query_embs = torch.load(PROCESSED / "query_embeddings.pt", weights_only=True)

    pcst_cfg = cfg["pcst"]
    embed_fn = lambda texts: embed_texts(
        texts, model_name=cfg["embedding"]["model"], batch_size=cfg["embedding"]["batch_size"]
    )

    rows = []
    for min_shared in [1, 2, 3]:
        builder = EntityGraphBuilder(min_shared_entities=min_shared)
        data, tn, te = builder.build(corpus, embeddings, entities, edge_embedder=embed_fn)

        results = []
        for i, q in enumerate(queries):
            retrieved = retrieve_with_pcst(
                query_embs[i], data, tn, te,
                topk=pcst_cfg["topk"], topk_e=pcst_cfg["topk_e"], cost_e=pcst_cfg["cost_e"],
            )
            results.append({
                "retrieved_doc_ids": retrieved,
                "gold_doc_ids": q["gold_doc_ids"],
                "question_type": q["question_type"],
            })

        m = aggregate(results)
        rows.append({"min_shared_entities": min_shared, "n_edges": data.edge_index.shape[1] // 2, **m})
        logger.info("min_shared=%d -> F1=%.4f recall=%.4f edges=%d",
                    min_shared, m["f1"], m["evidence_recall"], data.edge_index.shape[1] // 2)

    return rows


def _error_analysis(queries):
    from src.data.loader import load_jsonl
    from src.evaluation.metrics import evidence_recall

    combined_path = RETRIEVAL / "combined_pcst.jsonl"
    entity_path = RETRIEVAL / "entity_pcst.jsonl"
    semantic_path = RETRIEVAL / "semantic_pcst.jsonl"

    if not (combined_path.exists() and entity_path.exists()):
        logger.warning("Retrieval results not found; skipping error analysis")
        return

    combined_results = {r["query_id"]: r for r in load_jsonl(combined_path)}
    entity_results = {r["query_id"]: r for r in load_jsonl(entity_path)}

    # Queries where combined recall > entity recall
    wins = []
    for qid, c_r in combined_results.items():
        e_r = entity_results.get(qid)
        if e_r is None:
            continue
        c_recall = evidence_recall(c_r["retrieved_doc_ids"], c_r["gold_doc_ids"])
        e_recall = evidence_recall(e_r["retrieved_doc_ids"], e_r["gold_doc_ids"])
        if c_recall > e_recall:
            wins.append({
                "query_id": qid,
                "query": c_r["query"],
                "combined_recall": c_recall,
                "entity_recall": e_recall,
                "gold_doc_ids": c_r["gold_doc_ids"],
                "combined_retrieved": c_r["retrieved_doc_ids"],
                "entity_retrieved": e_r["retrieved_doc_ids"],
            })

    wins = sorted(wins, key=lambda x: x["combined_recall"] - x["entity_recall"], reverse=True)[:20]
    _write_csv(wins, METRICS / "error_analysis_combined_beats_entity.csv")
    logger.info("Error analysis: %d queries where combined > entity (top-20 saved)", len(wins))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
