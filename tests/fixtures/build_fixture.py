"""Cuts the small test dataset out of the full MultiHop-RAG dataset.

WHY THIS EXISTS
---------------
The smoke test needs a corpus small enough to run in seconds. It cannot just
take the first 40 documents of MultiHop-RAG, because a query's gold evidence
would then mostly point at documents that are no longer there, and the test
would have nothing to retrieve.

So this script picks the queries first, keeps every document those queries
need, and pads up to the requested size with random distractors. It also
renumbers doc_id, because the whole pipeline uses doc_id as a graph node index:
document number 7 must be row 7 of the embedding matrix and node 7 of the graph.

The output is two small JSONL files that are committed to the repo. You do NOT
need to run this - the smoke test just reads those files, and works offline.

    python tests/fixtures/build_fixture.py                     # 8 queries, 40 docs
    python tests/fixtures/build_fixture.py --queries 20        # 20 queries, 100 docs
    python tests/fixtures/build_fixture.py --queries 20 --docs 150

HOW MANY DOCUMENTS FOR HOW MANY QUERIES?
----------------------------------------
Rule of thumb: **documents = 5 x queries** (which is where 8 -> 40 comes from).
That is what --docs defaults to, so normally you only pass --queries.

The reasoning: each query here has 2-4 gold documents, and they overlap a
little between queries, so N queries need roughly 2.5N distinct gold documents.
At 5N documents, gold is about half the corpus and the other half is
distractors - enough for retrieval to make mistakes worth looking at.

Going lower makes the test dishonest: at 3N the corpus is almost all gold
documents, so any retrieval method scores well and the test stops telling you
anything. Going higher is more realistic but gets slow fast, because the number
of edges to embed grows roughly with docs^2:

    queries  docs   test runtime   peak memory
          8    40      50 s  (measured)   2 MB
         12    60     110 s  (measured)   6 MB
         20   100      ~5 min (estimate) 15 MB
         40   200     ~20 min (estimate) 60 MB

Memory is dominated by SemanticGraphBuilder, which materialises a full
docs x docs x embedding_dim tensor (docs^2 x 384 x 4 bytes at MiniLM size).
The full 609-document corpus would need ~570 MB for that one step alone.
Above ~100 documents you are no longer running a smoke test - use
scripts/01..04 instead.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_multihop_rag, save_jsonl

HERE = Path(__file__).resolve().parent
DOCS_PER_QUERY = 5  # the rule of thumb above
SEED = 20260727


def main():
    args = _parse_args()
    n_queries = args.queries
    n_docs = args.docs or n_queries * DOCS_PER_QUERY

    # The real loader, so the fixture is parsed exactly like the real pipeline
    # parses it (including resolving evidence titles to document ids).
    corpus, queries = load_multihop_rag()
    print(f"full dataset: {len(corpus)} docs, {len(queries)} queries")
    print(f"cutting a slice of {n_queries} queries and {n_docs} docs")

    rng = random.Random(args.seed)

    # 1. Pick queries that have 2-4 distinct gold documents. Fewer than 2 is
    #    not multi-hop; more than 4 would eat too much of a small corpus.
    eligible = [q for q in queries if 2 <= len(set(q["gold_doc_ids"])) <= 4]
    if n_queries > len(eligible):
        sys.exit(f"only {len(eligible)} queries have 2-4 gold docs, cannot pick {n_queries}")
    picked = rng.sample(eligible, n_queries)

    # 2. Keep every gold document, then pad with random distractors so that
    #    retrieval has wrong answers available to pick.
    gold = sorted({d for q in picked for d in q["gold_doc_ids"]})
    if len(gold) > n_docs:
        sys.exit(
            f"{n_queries} queries need {len(gold)} gold docs, more than --docs {n_docs}. "
            f"Raise --docs (the rule of thumb suggests {n_queries * DOCS_PER_QUERY})."
        )
    others = [i for i in range(len(corpus)) if i not in set(gold)]
    keep = sorted(gold + rng.sample(others, n_docs - len(gold)))
    print(f"keeping {len(gold)} gold docs + {n_docs - len(gold)} distractors "
          f"({len(gold) / n_docs:.0%} of the corpus is gold)")

    # 3. Renumber: old document id -> new position 0..n_docs-1.
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
