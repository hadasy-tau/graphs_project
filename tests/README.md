# Pipeline smoke test

`test_pipeline_smoke.py` runs the whole pipeline — load, NER, embed, build all
four graphs, retrieve, evaluate — on 40 documents and 8 queries, in under a
minute. It exists so you can watch the flow and see what each stage produces.

## Run it

```bash
python tests/test_pipeline_smoke.py
```

```bash
python -m pytest tests/test_pipeline_smoke.py -s
```

In VS Code, `.vscode/launch.json` has the configurations ready — pick
**"Smoke test: full pipeline (debug)"** and press F5. `justMyCode` is off, so
you can also step into PyG's `retrieval_via_pcst`.

Results go to `data/smoke_run/`: the four graphs, one JSONL per retrieval
condition, and `metrics/summary_table.csv`. Document embeddings are cached
there too — `rm -rf data/smoke_run` to start clean.

## The three files

| File | What it is |
|---|---|
| `test_pipeline_smoke.py` | The test. One function, top to bottom, in the order `scripts/01..04` run. |
| `fixtures/*.jsonl` | The 40-doc / 8-query dataset it reads. Committed, so the test works offline. |
| `fixtures/build_fixture.py` | How those JSONL files were made. You don't need to run it. |
| `pcst_fallback.py` | Compatibility shim, see below. You don't need to read it. |

### Where to put breakpoints

Everything is one function, so any breakpoint gives you every variable computed
so far — `corpus`, `entities`, `doc_embs`, `graphs`, `results`. Interesting
spots are the section banners: `=== 2. preprocess ===` to see what spaCy
extracted, `=== 4. build graphs ===` for the edge lists, `=== 6. retrieval ===`
to compare gold against retrieved for one query.

### Why `build_fixture.py` exists

The test needs a small corpus, but you cannot just take the first 40 documents
of MultiHop-RAG: the queries' gold evidence would point at documents that are
no longer there, leaving nothing to retrieve. So the script picks 8 queries
first, keeps all the documents they need, pads to 40 with distractors, and
renumbers `doc_id` — because the pipeline uses `doc_id` as a graph node index,
so document 7 has to be row 7 of the embedding matrix and node 7 of the graph.

Run it only if you want a different or bigger slice.

## Two things are scaled down

The test is tuned so the flow is **visible**, not so the numbers are
publishable. Don't draw conclusions from a smoke-test run.

**1. Smaller models.** `config/test_small.yaml` uses `all-MiniLM-L6-v2` instead
of `bge-large-en-v1.5` and `en_core_web_sm` instead of `en_core_web_lg`, and
drops the semantic threshold from 0.75 to 0.40 because MiniLM's similarity
scale is much lower — that file explains each choice. To run against the real
config, change the filename in the test's `yaml.safe_load(...)` line.

**2. The PCST solver.** `pcst-fast` ships C++ source with no prebuilt wheels,
so installing it needs a compiler — on Windows, Microsoft C++ Build Tools
14.0+. When it's missing, `pcst_fallback.py` substitutes a simple greedy
heuristic so the PCST stage still runs. The test prints which one is active on
its first line. Install the real one before running
`scripts/03_run_retrieval.py` for actual results:

```bash
pip install pcst-fast
```
