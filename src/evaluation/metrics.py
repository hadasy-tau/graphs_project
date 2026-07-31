"""Retrieval evaluation metrics for GraphRAG evidence retrieval."""
from __future__ import annotations

from statistics import mean


def evidence_recall(retrieved: list[int], gold: list[int]) -> float:
    """Fraction of distinct gold docs that appear in retrieved set.

    MultiHop-RAG sometimes cites the same article multiple times for different
    evidence facts. Retrieval is document-level, so duplicate gold document ids
    should not reduce recall when the document was retrieved.
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    return len(set(retrieved) & gold_set) / len(gold_set)


def full_hit_rate(retrieved: list[int], gold: list[int]) -> bool:
    """True iff ALL gold docs are in retrieved set."""
    return set(gold).issubset(set(retrieved))


def precision(retrieved: list[int], gold: list[int]) -> float:
    """Fraction of retrieved docs that are gold."""
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(retrieved)


def f1(retrieved: list[int], gold: list[int]) -> float:
    r = evidence_recall(retrieved, gold)
    p = precision(retrieved, gold)
    if r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-query results into mean metrics.

    Each result dict must have keys: retrieved_doc_ids, gold_doc_ids.
    """
    recalls, hits, precs, f1s, sizes = [], [], [], [], []
    for r in results:
        ret = r["retrieved_doc_ids"]
        gold = r["gold_doc_ids"]
        recalls.append(evidence_recall(ret, gold))
        hits.append(float(full_hit_rate(ret, gold)))
        precs.append(precision(ret, gold))
        f1s.append(f1(ret, gold))
        sizes.append(len(ret))

    return {
        "evidence_recall": round(mean(recalls), 4),
        "full_hit_rate": round(mean(hits), 4),
        "precision": round(mean(precs), 4),
        "f1": round(mean(f1s), 4),
        "avg_retrieved_size": round(mean(sizes), 2),
        "n_queries": len(results),
    }


def aggregate_by_qtype(results: list[dict]) -> dict[str, dict]:
    """Aggregate metrics broken down by question_type field."""
    by_type: dict[str, list[dict]] = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        by_type.setdefault(qtype, []).append(r)
    return {qtype: aggregate(recs) for qtype, recs in by_type.items()}
