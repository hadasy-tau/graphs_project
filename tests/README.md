# Tests

Two tests with different jobs.

| | |
|---|---|
| `test_pipeline_smoke.py` | Does the pipeline run end to end? Tiny data, scaled-down models, under a minute. Says nothing about graph quality. |
| `test_build_graphs.py` | Builds the four graph variants and dumps them to CSV so you can study them. Runs on the **full corpus with the real models**. |

Plus three supporting files you rarely need to open: `common.py` (dataset
subsetting and logging setup, shared by both tests and the fixture builder),
`pcst_fallback.py`, and `fixtures/`.

---

# 1. Pipeline smoke test

Runs the whole pipeline — load, NER, embed, build all four graphs, retrieve,
evaluate — on 40 documents and 8 queries. It exists so you can watch the flow
and see what each stage produces.

```bash
python tests/test_pipeline_smoke.py
```

```bash
python -m pytest tests/test_pipeline_smoke.py -s
```

To debug it in VS Code, open the file and press F5. Turn off
**Python › Debugging: Just My Code** in settings first, otherwise the debugger
refuses to step into library code and you can't see inside PyG's
`retrieval_via_pcst` — the actual G-Retriever subgraph selection.

Results go to `tests/results/smoke_run/`. Everything is one function, so a breakpoint
anywhere gives you every variable computed so far — `corpus`, `entities`,
`doc_embs`, `graphs`, `results`. Interesting spots are the section banners:
`=== 2. preprocess ===` for what spaCy extracted, `=== 6. retrieval ===` to
compare gold against retrieved.

### Its dataset

`fixtures/*.jsonl` holds 40 documents and 8 queries, committed so the test runs
offline. `fixtures/build_fixture.py` made them — you don't need to run it.

It exists because you cannot just take the first 40 documents of MultiHop-RAG:
the queries' gold evidence would point at documents that are no longer there,
leaving nothing to retrieve. So it picks queries first, keeps all the documents
they need, pads with distractors, and renumbers `doc_id` — the pipeline uses
`doc_id` as a graph node index, so document 7 has to be row 7 of the embedding
matrix and node 7 of the graph.

That selection logic lives in `common.py` as `subset_corpus`, because
`test_build_graphs.py --docs N` needs exactly the same thing at runtime.
`build_fixture.py` is just a CLI around it that writes the result to JSONL.

To cut a different slice (the rule of thumb is 5 documents per query, which is
where 8 → 40 comes from; the file explains the reasoning and the cost):

```bash
python tests/fixtures/build_fixture.py --queries 20
```

### Two things are scaled down

The smoke test is tuned so the flow is **visible**, not so the numbers mean
anything. Use `test_build_graphs.py` for graphs you intend to reason about.

**Smaller models.** `config/test_small.yaml` uses `all-MiniLM-L6-v2` instead of
`bge-large-en-v1.5` and `en_core_web_sm` instead of `en_core_web_lg`, and drops
the semantic threshold from 0.75 to 0.40 because MiniLM's similarity scale is
much lower. That file explains each choice.

**The PCST solver.** `pcst-fast` ships C++ source with no prebuilt wheels, so
installing it needs a compiler — on Windows, Microsoft C++ Build Tools 14.0+.
When it's missing, `pcst_fallback.py` substitutes a greedy heuristic so the
stage still runs. The test prints which one is active on its first line.
Install the real one before running `scripts/03_run_retrieval.py`:

```bash
pip install pcst-fast
```

---

# 2. Graph construction test

Builds the four graph variants and writes them out as CSV. No retrieval, no
PCST, no metrics — just the graphs, in a form you can open in Excel.

```bash
python tests/test_build_graphs.py
```

That is the **full 609-document corpus with the real `base.yaml` models**, so
the graphs are exactly the ones your experiments will use. Roughly 4 minutes
the first time on CPU; embeddings and NER are cached, so re-runs are seconds.

```bash
python tests/test_build_graphs.py --docs 100      # faster, but see the warning
python tests/test_build_graphs.py --config config/test_small.yaml --docs 60
python -m pytest tests/test_build_graphs.py       # quick structural check
```

### Why the full corpus is affordable

A graph's structure depends only on the edge **rules**. The 1024-dim
`edge_attr` vectors that `GraphBuilder.build()` computes are used later by PCST
for scoring and change no edge at all. Embedding tens of thousands of edge
texts is the expensive part of `scripts/02`, so this test calls `get_edges()`
directly and skips it.

### What `--docs` costs you

Only the metadata graph survives subsetting intact:

| graph | on a subset |
|---|---|
| metadata | **exact** — the rules are pairwise and absolute, so you get precisely the induced subgraph |
| entity | **approximate** — `stop_entity_threshold` is a *fraction* of the corpus, so 100 docs drops entities appearing in >20 of them while 609 drops >122. Canonicalization also clusters over whatever vocabulary is present |
| semantic | **approximate** — pair similarities don't move, but mutual k-NN is global: your top-6 neighbours among 100 documents aren't your top-6 among 609 |
| combined | **approximate**, being the union of the three |

Use `--docs` to iterate, the default for anything you'll draw a conclusion
from. The test prints which mode it ran in.

### What it writes

Into `tests/results/graph_runs/<config>_<n>docs/`, all keyed by `doc_id`:

| file | contents |
|---|---|
| `documents.csv` | what is node 7 — title, source, category, date, and its entities |
| `queries.csv` | the questions, with gold doc ids **and gold titles** spelled out |
| `graphs/<name>_edges.csv` | every edge and *why* it exists (the edge text) |
| `graphs/<name>_adjacency.csv` | the graph as a `doc_id × doc_id` 0/1 matrix |
| `graph_stats.csv` | nodes, edges, avg degree, components, oracle connectivity |
| `gold_connectivity.csv` | **can this graph reach each query's evidence?** |

All four adjacency matrices use the same node order, so subtracting two of them
shows exactly which edges one strategy adds over another.

`gold_connectivity.csv` is the one to look at first — it answers the project's
actual question, per query and per graph:

- `direct_edges` — gold pairs joined by a single edge, what a 1-hop retriever can follow
- `pairs_reachable` — gold pairs connected by any path; **below `n_gold_pairs` means the evidence sits in separate components and no amount of graph walking will join it**
- `max_hops` — how far a retriever would have to walk (blank if not all pairs are reachable)
