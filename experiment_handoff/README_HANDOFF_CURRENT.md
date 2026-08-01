# Current Experiment Handoff

This is the consolidated handoff folder after the k=17 random/null baseline run.

## Start here

1. `random_baselines/RANDOM_BASELINES_ADDENDUM.md`  
   Post-random update, key results, and current interpretation.

2. `random_baselines/k17_nulls_exact_metrics_with_hit_counts.csv`  
   Exact metrics, including average gold documents found per condition.

3. `random_baselines/summary_table.csv`  
   Standard k17_nulls summary table with 25 conditions.

4. `experiment_decisions_and_limitations.md`  
   Original decision/limitations document, now with a post-random update note at the top.

## Important status

The older pre-random materials are preserved for provenance, but any statement saying that random baselines are “not yet established” is stale.

The random/null baseline experiment has now been run and verified:

- 8 null graphs were generated with seed 123.
- 28 null graph files were verified.
- `k17_nulls` produced 25 retrieval JSONLs.
- `k17_nulls` produced 25 summary rows.
- The 9 copied real/dense conditions are identical to `mutual_knn_k17`.

## Current high-level conclusion

The final story should not be “semantic is best.”

A better supported story is:

> Differences between real graph construction strategies are smaller than they first appeared once retrieved size is controlled, especially for precision/F1. However, real graph structure itself carries substantial retrieval signal: replacing real edges with random controls sharply reduces evidence recovery.

## Still pending

- Bootstrap confidence intervals.
- Edge-prize/template-confound diagnostic.
- Final paper tables and figures.
- Clean Git commit from a clean clone.

## Safety

Do not use `--force-graphs`, `--prune-graphs`, Run All, or `git add .`.
