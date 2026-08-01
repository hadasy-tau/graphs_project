# Random Baseline Addendum

This folder is the consolidated handoff after running the k=17 random/null baselines.
It supersedes the previous `experiment_handoff_pre_random` status for any claims about random graph controls.

## Run status

- Arm: `k17_nulls`
- Regime: `mutual_knn_k=17`
- Null flag: `null_baselines=true`
- Seed for null graph generation: `123`
- Retrieval JSONLs: `25 / 25`
- Summary rows: `25 / 25`
- Saved null graph files: `28 / 28`
- The 9 copied real/dense conditions are bit-identical to `mutual_knn_k17` metrics.

## Main interpretation

The random baselines answer a different control question from the construction-strategy comparison.

The construction-strategy question asks whether entity, metadata, semantic, and combined graph construction differ under the same retrieval pipeline.

The random-control question asks whether real graph structure carries retrieval signal at all, or whether a random graph with comparable size/connectivity would behave similarly.

The current evidence supports the following cautious conclusion:

> Differences between real graph construction strategies are modest after retrieved-size control, especially for precision/F1. However, real graph structure itself carries substantial retrieval signal: replacing real edges with random controls sharply reduces evidence recovery.

## Clean no-PCST comparison

The no-PCST entity comparison is nearly size matched:

| condition | avg retrieved size | evidence recall | full-hit | avg gold found |
|---|---:|---:|---:|---:|
| entity_no_pcst | 12.79 | 0.7614 | 0.4643 | 1.982 |
| entity_random_structure_no_pcst | 12.90 | 0.3957 | 0.1175 | 0.969 |

At essentially the same retrieval budget, the real graph recovers about twice as many gold documents and has roughly four times the full-hit rate.

## PCST comparison

Real PCST graphs:
- avg retrieved size range: 9.18–11.75
- avg gold found range: 1.742–1.810

Null PCST graphs:
- avg retrieved size range: 19.53–23.50
- avg gold found range: 1.453–1.635

The PCST nulls retrieve roughly twice as many documents, yet still recover fewer gold documents on average. Therefore, direct recall comparison remains size-confounded, but the direction of the size confound favors the nulls, not the real graphs.

## What changed relative to the pre-random handoff

Any statement saying that random baselines are “not yet established” is now stale.

The following claim is now supported, with the caveat that this is based on the k=17 random-control arm:

> Real graph structure outperforms random graph controls for evidence recovery. In the no-PCST setting this is shown at matched retrieved size; in the PCST setting the nulls retrieve many more documents but still recover less evidence.

This does not mean that one real graph construction strategy is universally best. The safer interpretation is:

> Specific construction choices have modest size-controlled differences, while real-vs-random structure differences are large.

## Still pending

- Bootstrap confidence intervals for key matched comparisons.
- Edge-prize distribution diagnostic to address the edge-template confound.
- Final paper tables/figures.
- A clean Git commit from a clean clone. Do not commit from the current symlinked Colab state.

## Safety notes

Do not use:
- `--force-graphs`
- `--prune-graphs`
- Run All
- `git add .`

The null graph files now live inside `data/graphs/g-c27614a6ba`.
