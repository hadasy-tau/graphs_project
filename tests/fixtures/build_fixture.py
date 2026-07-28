"""Writes the small dataset that test_pipeline_smoke.py reads.

The smoke test needs a corpus small enough to run in seconds, and it should not
need a network connection. So this script cuts a slice out of MultiHop-RAG once
and saves it as two JSONL files, which are committed to the repo.

You do NOT need to run this. Run it only to cut a different slice:

    python tests/fixtures/build_fixture.py                 # 8 queries, 40 docs
    python tests/fixtures/build_fixture.py --queries 20    # 20 queries, 100 docs
    python tests/fixtures/build_fixture.py --queries 20 --docs 150

The selection logic lives in tests/common.py (subset_corpus), which
test_build_graphs.py uses too - it explains why queries are picked before
documents and why doc_id has to be renumbered.

HOW MANY DOCUMENTS FOR HOW MANY QUERIES?
----------------------------------------
Rule of thumb: documents = 5 x queries (which is where 8 -> 40 comes from).
That is what --docs defaults to, so normally you only pass --queries.

Each query has 2-4 gold documents and they overlap a little between queries, so
N queries need roughly 2.5N distinct gold documents. At 5N, gold is about half
the corpus and the rest is distractors - enough for retrieval to make mistakes
worth looking at. Go much lower and the corpus is almost all gold documents, so
any retrieval method scores well and the test stops telling you anything.

Going higher is more realistic but slower, because the number of edges to embed
grows roughly with docs^2:

    queries  docs   smoke test runtime
          8    40      50 s  (measured)
         12    60     110 s  (measured)
         20   100      ~5 min (estimate)

Above ~100 documents you are not running a smoke test any more. Use
tests/test_build_graphs.py to look at graphs at full corpus size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from common import DOCS_PER_QUERY, SEED, subset_corpus
from src.data.loader import load_multihop_rag, save_jsonl

HERE = Path(__file__).resolve().parent


def main():
    args = _parse_args()
    n_docs = args.docs or args.queries * DOCS_PER_QUERY

    # The real loader, so the fixture is parsed exactly like the real pipeline
    # parses it (including resolving evidence titles to document ids).
    corpus, queries = load_multihop_rag()
    print(f"full dataset: {len(corpus)} docs, {len(queries)} queries")
    print(f"cutting a slice of {args.queries} queries and {n_docs} docs")

    mini_corpus, mini_queries = subset_corpus(
        corpus, queries, n_docs=n_docs, n_queries=args.queries, seed=args.seed)

    # subset_corpus drops queries whose gold evidence will not fit; for a
    # committed fixture we would rather fail than silently write a smaller one.
    if len(mini_queries) < args.queries:
        sys.exit(
            f"only {len(mini_queries)} of {args.queries} queries fit in {n_docs} docs. "
            f"Raise --docs (the rule of thumb suggests {args.queries * DOCS_PER_QUERY})."
        )

    save_jsonl(mini_corpus, HERE / "mini_corpus.jsonl")
    save_jsonl(mini_queries, HERE / "mini_queries.jsonl")

    n_gold = len({d for q in mini_queries for d in q["gold_doc_ids"]})
    print(f"{n_gold} gold docs + {n_docs - n_gold} distractors "
          f"({n_gold / n_docs:.0%} of the corpus is gold)")
    print(f"wrote {len(mini_corpus)} docs and {len(mini_queries)} queries to {HERE}")
    for q in mini_queries:
        print(f"  q{q['query_id']} [{q['question_type']:<16}] gold={q['gold_doc_ids']}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=int, default=8, help="how many queries (default 8)")
    p.add_argument("--docs", type=int, default=None,
                   help=f"how many documents (default: queries x {DOCS_PER_QUERY})")
    p.add_argument("--seed", type=int, default=SEED, help="pick a different random slice")
    return p.parse_args()


if __name__ == "__main__":
    main()
