# Handoff Before Random Baselines

Date: 2026-08-01  
Status: completed controlled graph-construction experiments and PCST sensitivity; random graph baselines remain next.

## Read this first: do not use legacy repo-level results

The following repo-level paths are legacy flat-layout outputs from an older pipeline/config and should not be used for current paper numbers:

- `results/metrics/`
- `results/retrieval/`
- `results2/`

Use the files inside this handoff folder instead.

## What this folder is

This folder is a self-contained handoff package for a teammate who may start analysing the completed experiments or drafting the paper before the random-baseline experiment is complete.

For methodological details, claim wording, limitations, and what not to overclaim, read:

`experiment_decisions_and_limitations.md`

## Provenance

Completed arms were produced at git SHA `0478e919`.

Important seed decision for the next experiment: use `seed=123`, not `seed=42`.

## Completed arms included here

- `baseline`
- `mutual_knn_k10`
- `mutual_knn_k17`
- `mutual_knn_k25`
- `k17_pcst_topk3`
- `k17_pcst_topk9`
- `k17_pcst_cost025`
- `k17_pcst_cost10`

Each arm is stored under:

`experiment_handoff_pre_random/arms/<arm>/`

Each arm includes:

- `config.yaml`
- `manifest.json`
- `metrics/summary_table.csv`
- `metrics/by_qtype_table.csv`
- `metrics/graph_stats.csv`

The `metrics/` folder is the canonical corrected folder after the gold-dedup fix.

If present, `metrics_pre_gold_dedup/` is archived only. Do not use it for paper numbers.

## Metric correction

A correctness issue was fixed after the original evaluations:

- 168 evaluable queries had duplicate `gold_doc_ids`.
- Retrieval is document-level, so recall should use distinct gold documents.
- `evidence_recall` now dedupes gold documents in the denominator.
- `oracle_connectivity` now dedupes gold documents before forming gold pairs.
- Precision, full-hit rate, retrieved size, and counts were unchanged.
- Recall increased uniformly by about +0.016 to +0.018.
- All pre-correction recall, F1, and oracle values are superseded.

Canonical corrected outputs in this handoff are under:

`experiment_handoff_pre_random/arms/<arm>/metrics/`

## Corrected analysis files

Current corrected PCST-sensitivity analysis files:

- `experiment_handoff_pre_random/analysis/pcst_sensitivity_k17_gold_dedup.csv`
- `experiment_handoff_pre_random/analysis/pcst_sensitivity_interpolated_by_size_gold_dedup.csv`

Code patch backup:

- `experiment_handoff_pre_random/analysis/code_patches/local_fixes_gold_dedup_and_option_c.patch`

## Current interim interpretation

These are interim findings, pending random graph baselines.

- At default PCST settings, semantic graph construction gives the most precise and compact retrieval, at the cost of recall.
- After size control in the k=17 PCST sweep, precision/F1 differences largely collapse.
- Recall and full-hit differences persist more than precision/F1 differences.
- At matched size 10 in the corrected k=17 sweep, recall ordering is:
  combined > semantic > entity > metadata.
- This matches the corrected k=17 oracle-connectivity ordering:
  combined > semantic > entity > metadata.
- Phrase cautiously: this is consistent with structural reachability constraining evidence recovery, not proof of a general law.

Measured near-matched comparisons useful for writing:

- `combined(default)` vs `entity(cost10)`, both size 9.89:
  combined +0.0256 recall, +0.0341 full-hit, precision within 0.0016.
- `semantic(cost025)` size 9.70 vs `metadata(default)` size 9.71:
  identical precision 0.1966, semantic +0.0209 recall, +0.0400 full-hit.

## Raw per-query outputs

Per-query retrieval JSONLs are not committed in this handoff. They remain on Drive.

Drive root:

`/content/drive/MyDrive/graphrag_mknn_2026_07_30`

Expected raw retrieval location per completed arm:

`results/<arm>/retrieval/`

Expected shape:

- 9 JSONL files per completed standard arm
- 2556 rows per JSONL

These raw JSONLs are required for error analysis, bootstrap, or per-query inspection. The repo is not self-sufficient for those tasks.

## Next experiment

Random graph baselines are the next major experiment.

Planned setup:

- New arm: `k17_nulls`
- k=17
- default PCST settings
- seed=123, not 42
- nulls:
  - `random_structure`
  - `shuffled_nodes`
- run both PCST and no-PCST
- compare recall size-aware using interpolation if null retrieved sizes differ

Required wiring edits still to do:

1. Add null graph names to `scripts/03_run_retrieval.py`.
2. Make evaluator expected conditions scoped/parameterized, not a global replacement.
3. Ensure `combined_cost_e` applies to `combined_random_structure` and `combined_shuffled_nodes`.

## Important seed note

Use `seed=123`.

Seed 42 produced an unusually disconnected shuffled-semantic graph: oracle 0.0034 versus density 0.0168. Using it would make the real graph beat an atypically weak null.
