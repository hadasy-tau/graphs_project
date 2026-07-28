"""Load and normalize the MultiHop-RAG dataset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_multihop_rag(
    cache_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (corpus_docs, queries) from MultiHop-RAG.

    corpus_docs keys: doc_id (int), title, body, source, published_at, category, url
    queries keys:     query_id (int), query, answer, question_type, gold_doc_ids (list[int])
    """
    from datasets import load_dataset

    # 1. Load the two configs separately
    corpus_dict = load_dataset("yixuantt/MultiHopRAG", "corpus", cache_dir=cache_dir)
    queries_dict = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", cache_dir=cache_dir)

    # 2. Extract the data splits
    corpus_split = corpus_dict["train"]
    queries_split = queries_dict["train"]

    # --- Build corpus from the corpus split ---
    corpus_docs = _build_corpus(corpus_split)

    # --- Build title → doc_id index for evidence resolution ---
    title_to_id: dict[str, int] = {}
    for doc in corpus_docs:
        key = _normalize_title(doc["title"])
        title_to_id[key] = doc["doc_id"]

    # --- Build queries from the queries split ---
    queries = _build_queries(queries_split, title_to_id)

    logger.info("Loaded %d corpus docs, %d queries", len(corpus_docs), len(queries))
    return corpus_docs, queries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    return title.lower().strip()


def _build_corpus(dataset) -> list[dict]:
    """Extract article records from the Hugging Face dataset split."""
    rows = []
    
    feature_names = set(dataset.features.keys())
    body_field = _pick_field(feature_names, ["body", "content", "text", "article"])
    title_field = _pick_field(feature_names, ["title", "headline"])

    if body_field is None or title_field is None:
        return _extract_corpus_from_nested(dataset)

    for i, row in enumerate(dataset):
        rows.append({
            "doc_id": i,
            "title": row.get(title_field, ""),
            "body": row.get(body_field, ""),
            "source": row.get("source", row.get("outlet", "")),
            "published_at": row.get("published_at", row.get("date", "")),
            "category": row.get("category", row.get("topic", "")),
            "url": row.get("url", ""),
            "author": row.get("author", row.get("authors", "")) or "",
        })
    return rows


def _build_queries(dataset, title_to_id: dict[str, int]) -> list[dict]:
    """Extract QA records and resolve evidence titles to doc_ids."""
    feature_names = set(dataset.features.keys())
    query_field = _pick_field(feature_names, ["query", "question"])
    answer_field = _pick_field(feature_names, ["answer", "answers", "gold_answer"])
    evidence_field = _pick_field(feature_names, ["evidence_list", "supporting_articles", "evidences"])
    qtype_field = _pick_field(feature_names, ["question_type", "type", "qtype"])

    queries = []
    unresolved_count = 0

    for i, row in enumerate(dataset):
        raw_evidences = row.get(evidence_field, []) if evidence_field else []
        gold_doc_ids, n_unresolved = _resolve_evidences(raw_evidences, title_to_id)
        unresolved_count += n_unresolved

        queries.append({
            "query_id": i,
            "query": row.get(query_field, ""),
            "answer": row.get(answer_field, ""),
            "question_type": row.get(qtype_field, "unknown"),
            "gold_doc_ids": gold_doc_ids,
        })

    if unresolved_count:
        logger.warning(
            "%d evidence titles could not be resolved to doc_ids (dataset quality issue)",
            unresolved_count,
        )
    return queries


def _resolve_evidences(
    raw_evidences: list[str] | list[dict],
    title_to_id: dict[str, int],
) -> tuple[list[int], int]:
    """Return (resolved_ids, num_unresolved)."""
    ids: list[int] = []
    unresolved = 0
    for ev in raw_evidences:
        # evidence_list may be a list of title strings or dicts with a 'title' key
        title = ev if isinstance(ev, str) else ev.get("title", "")
        key = _normalize_title(title)
        if key in title_to_id:
            ids.append(title_to_id[key])
        else:
            unresolved += 1
            logger.debug("Unresolved evidence title: %r", title)
    return ids, unresolved


def _pick_field(feature_names: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in feature_names:
            return c
    return None


def _extract_corpus_from_nested(dataset) -> list[dict]:
    """Last-resort extraction for non-standard HF dataset layouts."""
    raise ValueError(
        "Could not detect corpus columns in the MultiHop-RAG HF dataset. "
        "Please check the dataset schema and update loader.py accordingly."
    )


# ---------------------------------------------------------------------------
# Serialization helpers used by scripts
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]