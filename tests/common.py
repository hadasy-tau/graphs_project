"""Helpers shared by the two tests and by the fixture builder."""
from __future__ import annotations

import logging
import random
import sys

SEED = 20260727
DOCS_PER_QUERY = 5  # rule of thumb: N queries need about 5N documents


def setup_logging():
    """INFO logs from src/, without the download chatter. UTF-8 for Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "sentence_transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def subset_corpus(corpus, queries, n_docs=None, n_queries=None, seed=SEED):
    """Cut a smaller corpus that still contains its queries' gold evidence.

    You cannot just take the first N documents of MultiHop-RAG: the queries'
    gold evidence would point at documents that are no longer there, leaving
    nothing to retrieve. So queries are picked first, every document they need
    is kept, and the rest is padded with random distractors.

    Documents are renumbered 0..n-1 and each query's gold_doc_ids remapped,
    because the whole pipeline uses doc_id as a graph node index: document 7
    must be row 7 of the embedding matrix and node 7 of the graph.

    Args:
        corpus, queries: the full dataset, from load_multihop_rag().
        n_docs: how many documents to keep. None keeps the whole corpus.
        n_queries: how many queries to keep. None keeps as many as fit.
        seed: which random slice to take.

    Returns:
        (corpus, queries), renumbered. Fewer queries than requested come back
        if their gold evidence would not fit in n_docs - callers that care
        should check len() of the result.
    """
    if n_docs is None or n_docs >= len(corpus):
        keep = queries if n_queries is None else queries[:n_queries]
        return corpus, [q for q in keep if q["gold_doc_ids"]]

    rng = random.Random(seed)

    # Queries with 2-4 distinct gold documents. Fewer than 2 is not multi-hop;
    # more than 4 would eat too much of a small corpus.
    eligible = [q for q in queries if 2 <= len(set(q["gold_doc_ids"])) <= 4]
    picked = rng.sample(eligible, min(n_queries or len(eligible), len(eligible)))

    # Drop queries until their gold evidence fits in the corpus we are allowed.
    gold = sorted({d for q in picked for d in q["gold_doc_ids"]})
    while len(gold) > n_docs:
        picked.pop()
        gold = sorted({d for q in picked for d in q["gold_doc_ids"]})

    others = [i for i in range(len(corpus)) if i not in set(gold)]
    keep = sorted(gold + rng.sample(others, n_docs - len(gold)))
    new_id = {old: i for i, old in enumerate(keep)}

    sub_corpus = [{**corpus[old], "doc_id": i} for i, old in enumerate(keep)]
    sub_queries = [{**q, "query_id": i,
                    # set(): MultiHop-RAG sometimes cites the same article twice
                    "gold_doc_ids": sorted({new_id[d] for d in q["gold_doc_ids"]})}
                   for i, q in enumerate(picked)]
    return sub_corpus, sub_queries
