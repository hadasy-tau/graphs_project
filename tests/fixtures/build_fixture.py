"""Cuts the small test dataset out of the full MultiHop-RAG dataset.

WHY THIS EXISTS
---------------
The smoke test needs a corpus small enough to run in seconds. It cannot just
take the first 40 documents of MultiHop-RAG, because a query's gold evidence
would then mostly point at documents that are no longer there, and the test
would have nothing to retrieve.

So this script picks 8 queries first, keeps every document those queries need,
and pads up to 40 with random distractors. It also renumbers doc_id, because
the whole pipeline uses doc_id as a graph node index: document number 7 must be
row 7 of the embedding matrix and node 7 of the graph.

The output is two small JSONL files that are committed to the repo. You do NOT
need to run this - the smoke test just reads those files, and works offline.
Run it only if you want a different or larger slice:

    python tests/fixtures/build_fixture.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_multihop_rag, save_jsonl

HERE = Path(__file__).resolve().parent
N_DOCS = 40
N_QUERIES = 8
SEED = 20260727


def main():
    # The real loader, so the fixture is parsed exactly like the real pipeline
    # parses it (including resolving evidence titles to document ids).
    corpus, queries = load_multihop_rag()
    print(f"full dataset: {len(corpus)} docs, {len(queries)} queries")

    rng = random.Random(SEED)

    # 1. Pick queries that have 2-4 distinct gold documents. Fewer than 2 is
    #    not multi-hop; more than 4 would eat too much of a 40-doc corpus.
    eligible = [q for q in queries if 2 <= len(set(q["gold_doc_ids"])) <= 4]
    picked = rng.sample(eligible, N_QUERIES)

    # 2. Keep every gold document, then pad with random distractors so that
    #    retrieval has wrong answers available to pick.
    gold = sorted({d for q in picked for d in q["gold_doc_ids"]})
    others = [i for i in range(len(corpus)) if i not in set(gold)]
    keep = sorted(gold + rng.sample(others, N_DOCS - len(gold)))
    print(f"keeping {len(gold)} gold docs + {N_DOCS - len(gold)} distractors")

    # 3. Renumber: old document id -> new position 0..39.
    new_id = {old: i for i, old in enumerate(keep)}

    mini_corpus = [{**corpus[old], "doc_id": i} for i, old in enumerate(keep)]
    mini_queries = [{
        "query_id": i,
        "query": q["query"],
        "answer": q["answer"],
        "question_type": q["question_type"],
        # set(): MultiHop-RAG sometimes cites the same article twice
        "gold_doc_ids": sorted({new_id[d] for d in q["gold_doc_ids"]}),
    } for i, q in enumerate(picked)]

    save_jsonl(mini_corpus, HERE / "mini_corpus.jsonl")
    save_jsonl(mini_queries, HERE / "mini_queries.jsonl")

    print(f"wrote {len(mini_corpus)} docs and {len(mini_queries)} queries to {HERE}")
    for q in mini_queries:
        print(f"  q{q['query_id']} [{q['question_type']:<16}] gold={q['gold_doc_ids']}")


if __name__ == "__main__":
    main()
