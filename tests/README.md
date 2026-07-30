# Tests

One test: `test_pipeline_smoke.py` — does the pipeline run end to end? Tiny data,
scaled-down models, under a minute. It says nothing about graph quality.

Plus three supporting files you rarely need to open: `common.py` (dataset
subsetting and logging setup, shared with the fixture builder),
`pcst_fallback.py`, and `fixtures/`.

For real graphs and real numbers, run the pipeline itself —
`python scripts/experiment.py --id look --stages 01,02` — or a sweep from
`notebooks/colab_run_experiments.ipynb`.

---

# Pipeline smoke test

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

That selection logic lives in `common.py` as `subset_corpus`; `build_fixture.py`
is just a CLI around it that writes the result to JSONL.

To cut a different slice (the rule of thumb is 5 documents per query, which is
where 8 → 40 comes from; the file explains the reasoning and the cost):

```bash
python tests/fixtures/build_fixture.py --queries 20
```

### Two things are scaled down

The smoke test is tuned so the flow is **visible**, not so the numbers mean
anything. Run the real pipeline for graphs you intend to reason about.

**Smaller models.** `config/test_small.yaml` uses `all-MiniLM-L6-v2` instead of
`bge-large-en-v1.5` and `en_core_web_sm` instead of `en_core_web_lg`. Semantic
edges are mutual k-NN only (no absolute similarity threshold), so MiniLM's
lower cosine scale does not empty the graph. That file explains each choice.

**The PCST solver.** `pcst-fast` ships C++ source with no prebuilt wheels, so
installing it needs a compiler — on Windows, Microsoft C++ Build Tools 14.0+.
When it's missing, `pcst_fallback.py` substitutes a greedy heuristic so the
stage still runs. The test prints which one is active on its first line.
Install the real one before running `scripts/03_run_retrieval.py`:

```bash
pip install pcst-fast
```
