# Experiment Decisions and Limitations

<!-- POST_RANDOM_UPDATE_START -->
## Post-random update

The original version of this document was written before the random/null baseline run. The random-control question has now been answered for the k=17 regime.

### Random/null baseline status

- Arm: `k17_nulls`
- Null graph seed: `123`
- Retrieval JSONLs: `25 / 25`
- Summary rows: `25 / 25`
- Null graph files: `28 / 28`
- The 9 copied real/dense conditions are identical to `mutual_knn_k17`.

### Updated supported claim

Random baselines are no longer “not yet established.”

The current k=17 evidence supports the following claim:

> Real graph structure carries retrieval signal. Replacing real edges with random controls sharply reduces evidence recovery.

The cleanest comparison is the no-PCST entity pair:

| condition | avg retrieved size | evidence recall | full-hit | avg gold found |
|---|---:|---:|---:|---:|
| entity_no_pcst | 12.79 | 0.7614 | 0.4643 | 1.982 |
| entity_random_structure_no_pcst | 12.90 | 0.3957 | 0.1175 | 0.969 |

At essentially the same retrieved size, the real graph recovers about twice as many gold documents and has roughly four times the full-hit rate.

For PCST, the null graphs retrieve substantially more documents but still recover fewer gold documents on average:

- Real PCST avg retrieved size: 9.18–11.75
- Real PCST avg gold found: 1.742–1.810
- Null PCST avg retrieved size: 19.53–23.50
- Null PCST avg gold found: 1.453–1.635

### Updated interpretation

The random baselines should not be used mainly to rank entity vs metadata vs semantic vs combined. They answer a broader control question: whether meaningful graph structure matters at all compared with random connectivity.

Current interpretation:

> Specific graph construction choices have modest size-controlled differences, while real-vs-random structure differences are large.

### Still cautious / still pending

- This random-control result is currently established for the k=17 regime.
- Bootstrap intervals are still pending.
- Edge-prize/template-confound diagnostics are still pending.
- Do not claim that a single construction strategy is universally best.
<!-- POST_RANDOM_UPDATE_END -->



Working reference for the final paper: what we decided, why, and what we must not overclaim.

---

## 1. Research Goal

This project studies how different graph construction strategies affect evidence retrieval in a Multi-Hop RAG setting. The comparison is between practical graph representations induced by different edge construction strategies:

- Entity-based graph
- Metadata-based graph
- Semantic-similarity graph
- Combined graph

All variants use the same document nodes and the same downstream retrieval pipeline. The variable is how edges are constructed and represented.

This is a **system-level GraphRAG comparison**. We do not claim to isolate pure graph topology. Under PCST, edge attributes are part of the graph representation, because edge prizes depend on query–edge similarity. Differences between graph types may therefore reflect both topology and the textual representation of edges.

**Accurate framing:**
> We compare practical graph representations induced by entity-based, metadata-based, semantic-similarity-based, and combined graph construction strategies under a fixed retrieval pipeline.

**Not:**
> We isolate the pure topological effect of each edge type.

*(Note: the approved proposal uses the word "isolates". Soften it to "compares"/"contrasts" in the paper.)*

---

## 2. Graph Types

### Entity graph
Connects documents that share extracted entities. Intuition: documents mentioning the same important entities may provide complementary evidence for multi-hop questions.

Edge text encodes the shared entities themselves:
```
Shared entities: Apple, Tim Cook
```
Entity edges encode explicit symbolic overlap, but their edge text may be less query-similar than title-based descriptions.

### Metadata graph
Connects documents by similarity between metadata records (built from `metadata_graph.fields: [title, author]`), sparsified with mutual k-NN.

Edge text uses a title-based template:
```
Similar metadata record: {title_i} and {title_j}
```
**Treat carefully in the write-up.** The graph is built from metadata-record similarity, but the PCST edge text names article titles rather than describing the shared metadata relation. Metadata results therefore reflect both metadata-based topology and title-based edge representation.

### Semantic graph
Connects documents by cosine similarity between document embeddings, sparsified with mutual k-NN.

Edge text:
```
Semantically related: {title_i} and {title_j}
```
A natural representation for a semantic-neighbour relation, but it means semantic edges may receive strong PCST prizes whenever article titles are query-relevant.

### Combined graph
```
combined = entity ∪ metadata ∪ semantic
```
Useful for testing whether multiple edge signals complement one another. **It is never density-matched** to the components — being a union, it is structurally denser by construction (e.g. average degree 17.75 at k=17 versus ~8–10 for the components). This is a standing limitation for all combined-graph results.

---

## 3. Construction Parameters

Thresholds were chosen during preliminary calibration and sanity-check runs, then **fixed** before the final controlled comparisons, to avoid post-hoc tuning. The goal of calibration was not per-graph optimisation but avoiding obvious failure modes: overly sparse graphs, noisy hub-heavy graphs, excessive entity merging, or variants not comparable in scale.

### `canonicalize_threshold: 0.90`
Conservative merging of near-duplicate entity mentions while avoiding false merges between distinct entities. Raised from 0.85 after inspecting merge diagnostics: at 0.85, false merges such as `Arkansas/Kansas`, `AFC East/AFC West` and `ASEAN/Sean` appeared. At 0.90 those disappear while genuine variants (`ARKANSAS/Arkansas`, `AMC Theaters/Theatres`, `André/Andre Onana`) still merge. A deterministic normalisation pass (case, accents, punctuation, `@handles`) runs before fuzzy matching.

### `entity_graph.min_shared_entities: 2`
**Applies only to the baseline/null regime.** When `mutual_knn_k` is set to an integer, the entity builder overrides the floor to 1 and applies mutual k-NN instead:
```python
min_shared = self.min_shared_entities if self.mutual_knn_k is None else 1
```
So in the normalized regimes (k = 10/17/25) the entity graph uses `min_shared = 1` plus a k-NN cap. This is a second way the baseline regime differs from the normalized ones, on top of the missing degree cap.

### Semantic graph — no similarity threshold
The semantic builder is **pure mutual k-NN**; there is no similarity-threshold parameter in the current code or config. (An earlier version had `semantic_graph.threshold`; it was removed when the builder moved to the shared mutual-kNN sparsifier. Do not cite it.)

### `mutual_knn_k` — the density-normalisation mechanism
Varied explicitly: **k = 10, 17, 25** (sparse, middle, denser). Pinning k is what makes the "same conditions" comparison possible, because it gives entity, metadata and semantic the same degree cap.

Central regime: **k = 17**, chosen because it is the value the pipeline itself derives in the baseline/null setting when targeting entity's natural average degree — not a hand-picked number.

**Important:** mutual k-NN retains only ~50% of k as realized average degree (measured: metadata 0.48/0.49/0.50 and semantic 0.57/0.60/0.61 at k = 10/17/25). So k is *not* the achieved degree, and the auto-derivation in null mode undershoots its own target by roughly 2×. Describe k = 17 as the middle rung of a density ladder, **not** as "density-matched to the baseline entity graph."

We do not claim these values are globally optimal. A broader sweep is future work.

---

## 4. Experimental Regimes

Completed: `baseline/null`, `mutual_knn_k10`, `mutual_knn_k17`, `mutual_knn_k25`.

The **main controlled comparison** uses the normalized regimes (k = 10, 17, 25), where all three component graphs share a degree cap.

**baseline/null is diagnostic, not part of the controlled comparison.** It is mismatched on two dimensions at once — average degree spans 8.29–16.62 (2.0×) versus 1.23–1.26× in the normalized regimes, and max degree differs 89 vs 17 (5.2×). It also uses a different entity rule (`min_shared = 2`, uncapped).

Its diagnostic value is real and worth reporting: **among the component graphs** in the baseline/null regime, entity has the highest oracle connectivity after gold-dedup (**0.7335**; combined is higher at 0.8417 but is a union, not a component) — yet `entity_no_pcst` delivers the weakest retrieval relative to `dense_no_pcst` in that arm (**−0.103 recall, −0.158 full-hit**). This motivates the normalized regimes rather than merely excusing them.

**Validation:** the null arm derives k = 17, so the k=17 arm rebuilds metadata and semantic identically (2525/3105 edges, with identical PCST metrics). This confirms that the runs are consistent. For metadata and semantic PCST, baseline and k=17 are effectively identical. Differences in entity and combined mainly reflect the changed entity construction, while no-PCST cross-regime comparisons still remain affected by the retrieval-budget difference.

---

## 5. Retrieval Conditions

Per regime: dense retrieval (no graph), graph retrieval without PCST, graph retrieval with PCST.

- **Dense** is the graph-free, structure-agnostic baseline — it controls the no-graph setting.
- **No-PCST** is top-K seeds plus 1-hop graph expansion; it tests whether simple expansion helps.
- **PCST** selects a compact subgraph using node prizes and edge prizes.

**Because PCST requires a graph, dense retrieval cannot control the PCST setting.** This is the specific gap that random graph baselines fill.

### Control we already have
All conditions share the same documents, document embeddings, **node features (`Data.x`)**, query embeddings, retrieval algorithm and PCST hyperparameters. PCST node prizes are therefore *identical* across graph conditions; only topology and edge attributes vary. State this explicitly in Methods — but phrase the attribution as "topology **and edge attributes**", not "topology alone".

---

## 6. Evaluation Decisions

### Empty-gold queries excluded
Queries with no resolved gold evidence are excluded from all metrics. In the current data: **301 of 2556 excluded, 2255 evaluable**, identical across every condition.

Rationale: `evidence_recall` returns 1.0 and `full_hit_rate` returns True by vacuous truth on an empty gold set, handing every condition the same block of free perfect scores — inflating absolute numbers and, worse, *compressing the differences between conditions*, which is the signal we measure. Report the evaluable N alongside every table.

### Duplicate gold documents deduped (correctness fix)

**168 of the 2255 evaluable queries list the same document more than once** in `gold_doc_ids` — e.g. `[524, 354, 524, 169]`, `[360, 360, 36]`, `[367, 367, 367, 337]`. MultiHop-RAG cites one article for several evidence facts.

The original metrics mixed conventions, and the result was not merely imprecise but self-contradictory:

- `evidence_recall` used a **set** intersection in the numerator but `len(gold)` **with duplicates** in the denominator, so a perfect retriever scored 0.50–0.75 on these queries — recall could never reach 1.0.
- `full_hit_rate` used `set(gold).issubset(...)`, so it *could* reach 1.0. A query could therefore score `full_hit = 1.0` and `recall = 0.50` simultaneously, violating the invariant that you cannot fully hit but only partially recall.
- `oracle_connectivity` in `compute_stats` had the same flaw: duplicates formed self-pairs `(a, a)` — guaranteed misses, since the graphs have no self-loops — and double-counted cross-pairs.

Retrieval is document-level, so the correct denominator is the number of **distinct** gold documents. Both were corrected:

```python
# metrics.py
gold_set = set(gold);  return len(set(retrieved) & gold_set) / len(gold_set)
# base.py
gold = list(dict.fromkeys(q["gold_doc_ids"]))
```

Corroboration: `tests/common.py` already deduped when building the test fixture, with the comment *"set(): MultiHop-RAG sometimes cites the same article twice."* The production metric was simply inconsistent with the fixture convention. A title-collision check on the corpus found **0 collisions**, confirming these are genuine repeated citations rather than two articles collapsing onto one `doc_id`.

**Effect.** Precision, full-hit, retrieved size and all counts are **unchanged**. Recall rose by **+0.016 to +0.018** across arms and F1 followed. Oracle connectivity rose by +0.006 to +0.023, proportionally to each graph's existing rate. Crucially, **every between-condition delta moved by ≤ 0.003** — the same 168 queries appear in every condition, so levels shifted uniformly and comparisons were preserved.

**All pre-correction recall, F1 and oracle values are superseded.** Canonical outputs are in `results/<arm>/metrics/`. The original pre-correction outputs are archived as `results/<arm>/metrics_pre_gold_dedup/`.

### Retrieval budget differs across regimes
`dense_baseline.k` is auto-calibrated to entity-PCST's average output size, so it varies by regime:

| regime | budget K |
|---|---|
| baseline/null | 11 |
| k = 10 | 15 |
| k = 17 | 13 |
| k = 25 | 12 |

This confounds **cross-regime** comparisons of recall. It does **not** affect within-regime comparisons (graph vs dense inside one arm share a budget), which is where our main claims live.

**We can bound the confound empirically.** Metadata and semantic graphs are identical between baseline (K=11) and k=17 (K=13), so their no-PCST difference isolates budget alone: the graph-vs-dense gap shrinks ≈ **+0.008 per extra retrieved document**. Applying that to entity's deficit across baseline → k=10 (+0.101 total): roughly **32% attributable to budget, 68% to graph structure**. Report as a supported estimate, noting it assumes the per-document effect transfers across graph types.

---

## 7. Interim Findings from Completed Runs

> All numbers below are **post-gold-dedup**. Findings remain **interim** until the random graph baselines are complete.

### At default PCST settings

1. PCST produces **stable** graph-type rankings across regimes; no-PCST expansion does not.
2. Precision ordering under PCST is identical in all four regimes: **semantic > metadata > combined > entity**.
3. `semantic_pcst` ranks first in precision and F1 in all four regimes — but it also returns the **smallest** retrieved sets and ranks **third of four on recall** in every regime.
4. No graph condition beats dense retrieval on recall in any regime. The only positive full-hit deltas occur at k=10 and do **not** replicate.

**Critically, item 2 does not survive size control** — see §8. Default-setting rankings must never be presented as if they were matched-size rankings.

### After size control (PCST sensitivity, k = 17)

5. The precision gap between graph types **largely collapses**. Semantic − entity precision falls from **0.034** at default settings to **under 0.002** at matched size — a ~95% reduction. At size 10 all four types sit within a 0.003 band.
6. **Recall differences persist.** Recall spreads stay **10–20× larger** than precision spreads after size control (0.028–0.048 vs 0.002–0.003).
7. At matched size 10 the corrected recall ordering is **combined (0.7115) > semantic (0.7073) > entity (0.6855) > metadata (0.6833)**, which coincides with the corrected k=17 oracle-connectivity ordering **combined (0.6951) > semantic (0.5441) > entity (0.4869) > metadata (0.4576)**.
8. Measured near-matched pairs (no interpolation) support the same picture:
   - `combined(default)` vs `entity(cost10)`, both at size 9.89: combined **+0.0256 recall**, **+0.0341 full-hit**, precision within 0.0016.
   - `semantic(cost025)` 9.70 vs `metadata(default)` 9.71: **identical precision (0.1966)**, semantic **+0.0209 recall**, **+0.0400 full-hit**.

**Defensible claims:**
> In the completed default-PCST runs, semantic graph construction yielded the most precise and compact retrieval, at the cost of recall.

> Once retrieved size is controlled, precision differences between graph constructions largely disappear, while recall differences persist. At matched size in the k = 17 regime, recall ordering coincides with oracle-connectivity ordering, consistent with structural reachability constraining evidence recovery.

**Avoid:** "Semantic graph topology is simply better than entity graph topology."

### Not yet established

- Whether real graph structures **outperform random graph controls** under PCST → random baselines (§10). *The single most important open question.*
- Whether entity's lower PCST performance is partly explained by **edge-prize / template effects** → edge-prize diagnostic (§9, §11).
- Whether graph failures are due to **missing gold-evidence connectivity** or to **retrieval/scoring failure** → k-hop oracle connectivity (§11).
- Whether the recall/oracle correspondence is more than coincidence: it rests on **four graph types in one regime**, where a perfect rank match has ~4% probability by chance. k=10 and k=25 have only one PCST setting each, so no matched-size comparison is possible there. Report as *consistent with*, not *demonstrates*.

---

## 8. Retrieved-Size Confound

In all four regimes, the precision ranking is **exactly the inverse of the retrieved-size ranking**, position for position:

```
precision desc : semantic > metadata > combined > entity
size ascending : semantic < metadata < combined < entity
```

Precision is mechanically inverse to set size, so semantic's advantage may partly reduce to returning fewer documents.

PCST sensitivity on k=17 addressed this by analysing precision / recall / F1 **versus average retrieved size**, one curve per graph type.

**Result: for precision the curves largely collapse; for recall they do not.**

| metric | spread at size 9 | size 10 | size 11 | size 11.8 |
|---|---|---|---|---|
| precision | 0.0121 | **0.0031** | **0.0021** | **0.0022** |
| F1 | 0.0254 | 0.0053 | 0.0066 | 0.0075 |
| **recall** | 0.0724 | **0.0282** | **0.0385** | **0.0485** |

*(Values from `results/analysis/pcst_sensitivity_interpolated_by_size_gold_dedup.csv`; post-gold-dedup.)*

Semantic's margin over the *runner-up* on precision is only +0.0007 to +0.0011, and it actually **loses to metadata at size 9**. Its F1 lead decays to **+0.0004** at the top of the overlap band. Differences that small are not distinguishable from noise without confidence intervals, which we have not computed — so do not claim them.

Size ranges by graph type: semantic 6.91–11.89, metadata 7.24–12.57, combined 7.89–12.27, entity 8.68–14.97. **Common overlap 8.68–11.89**; comparisons outside that band are extrapolation.

Interpolated values are linear between measured sweep points. Prefer the measured near-matched pairs in §7 item 8 where possible — they need no interpolation at all.

### PCST sensitivity arms (on k = 17)
Default already covered by the `mutual_knn_k17` arm (`topk 6`, `cost_e 0.5`, `combined_cost_e 0.25`). Additional arms:

```
k17_pcst_topk3     pcst.topk = 3
k17_pcst_topk9     pcst.topk = 9
k17_pcst_cost025   pcst.cost_e = 0.25 , pcst.combined_cost_e = 0.125
k17_pcst_cost10    pcst.cost_e = 1.00 , pcst.combined_cost_e = 0.500
```

`combined_cost_e` is scaled proportionally so combined's density correction stays constant rather than drifting relative to the other graphs. `topk_e` is not varied — stated as a limitation.

---

## 9. Edge-Attribute / Edge-Prize Caveat

PCST edge prizes are `cos(query, edge_attr)`, and the templates differ by graph type:

```
entity   : shared entity names          (no titles)
metadata : "Similar metadata record: {title_i} and {title_j}"
semantic : "Semantically related: {title_i} and {title_j}"
```

Metadata and semantic use near-identical title-based templates; entity is the only one without titles. Since titles are far more query-similar than entity-name lists, entity's edges may be systematically under-prized for reasons of wording, not topology. At default settings the precision gaps are consistent with this: the two title-template graphs sit 0.007 apart while entity is 0.027 below them.

**Note the interaction with §8:** after size control that entity gap shrinks to under 0.002, so much of what looked like a template effect on *precision* was retrieved size. The template concern remains live for *recall*, where differences persist — and the edge-prize diagnostic (§11) is what would settle it.

This is **not necessarily a bug** — the project compares practical graph representations, and edge attributes are part of the representation. But it limits claims about topology alone.

**Limitation statement for the paper:**
> Because PCST assigns edge prizes using query–edge similarity, differences across graph types may reflect both graph topology and edge-text representation. In particular, entity edges are represented by shared entity names, while semantic and metadata edges use title-based templates. We therefore interpret the results as a comparison of practical graph representations rather than a pure topology-only ablation.

A template-controlled comparison is left for future work.

---

## 10. Random Graph Baselines (next major experiment)

Needed because dense retrieval controls the no-graph setting but **cannot** control PCST — PCST requires a graph. Planned on **k = 17, default PCST**:

1. **Random structure** — same node and edge counts, uniformly random pairings, recycling the original's `(edge_attr, edge text)` pairs so the prize distribution is held fixed. Tests whether the real graph beats random connectivity at matched density.
2. **Shuffled nodes** — the document↔structure correspondence is destroyed while every structural statistic is preserved exactly. Tests whether the alignment between structure and document content matters.

#### Option C: shuffled-nodes is implemented by permuting edges, not node content

The original implementation permuted `Data.x` and left `node_idx = range(n)`, so node index meant *position*, not document ID. Retrieval returns `node_idx`, and evaluation compares against `gold_doc_ids` — so positions would have been compared against document IDs, producing low-but-nonzero recall that **looks exactly like a successful null result**. The saved permutation was never applied anywhere.

The fix keeps every document at its own position and rewires the edges instead:

```python
new_edge_index = perm_t[data.edge_index]     # edge (u,v) -> (perm[u], perm[v])
# x, node_idx, textual_nodes all unchanged; edge_attr unchanged
```

This produces a valid shuffled-node null at the document-index level: document positions, node features, and `node_idx` remain aligned, while the edge topology is relabelled through the permutation. The graph therefore preserves the original structural statistics up to relabeling, but **destroys the original document–structure alignment**. The global edge-attribute / edge-prize distribution is preserved because `edge_attr` is kept in the same row order, but the original document pairs are intentionally *not* preserved — that is the point of the null.

Relative to the earlier `Data.x[perm]` formulation, Option C is equivalent at the document-graph level: both connect documents `perm[u]` and `perm[v]`, assign each document its own node prize, and attach `attr[k]` to edge *k*. So it changes the implementation, not the null's semantics — while removing the position-vs-document-ID hazard. (Equivalence holds up to PCST tie-breaking by index.)

Validated in memory on the k=17 semantic graph: `x`, `edge_attr`, `node_idx` and `textual_nodes` unchanged; `edge_index == perm[orig.edge_index]`; degree sequence and component sizes identical to the original; edge-set overlap with the original 0.0158 ≈ density 0.0168.

#### Seed choice: 123, not 42

A seed sweep on shuffled-semantic oracle connectivity (density = 0.0168) gave: seed 1 → 0.0179, seed 7 → 0.0291, **seed 42 → 0.0034**, seed 123 → 0.0172, seed 2026 → 0.0131. Seed 42 produced a shuffled graph ~5× more disconnected than a typical draw, which would have inflated the real graph's apparent advantage. **Use seed 123.** `random_baselines.py` defaults to 42, so this needs an explicit override.

**Degree-preserving rewiring is not planned** for k=17: the normalized graphs are degree-capped at 17, so the distribution is flat and rewiring collapses toward random-structure. It would only be informative in the unnormalized baseline regime (entity max degree 89).

Interpretation:
- Real ≫ both nulls → graph construction carries genuine signal.
- Real ≈ nulls → the conclusion must soften; PCST compactness or generic structure explains much of the performance. **This is a legitimate finding, not a failed experiment.**

**Size-aware comparison is mandatory.** The nulls produce their own PCST output sizes. If a null's size differs from the real graph's, compare against the real graph's recall *interpolated at the null's size* using the §8 sweep curves — otherwise the size confound is reintroduced.

**Implementation: three edits required.** `src/graph/random_baselines.py` generates the graphs but is not wired into the pipeline.
1. `graph_names` in `scripts/03_run_retrieval.py` (hardcoded at line 35).
2. `EXPECTED_CONDITIONS` in `src/evaluation/evaluator.py` — pass as a **parameter** rather than editing the constant, or re-evaluating the existing 9-condition arms will raise.
3. `combined_cost_e` is applied by exact name match (`pcst_cfg_by_graph["combined"]`), so `combined_random_structure` and `combined_shuffled_nodes` would run at `cost_e = 0.5` instead of `0.25` — non-comparable to real combined. Match on prefix or map explicitly.

Planned as a **new arm (`k17_nulls`)** rather than modifying `mutual_knn_k17`, to leave analysed artifacts untouched.

---

## 11. Cheap CPU-Only Diagnostics

- **Edge-prize distributions** — per-graph distribution of `cos(query, edge_attr)`; tests §9 directly. Needs only saved embeddings, no retrieval, no GPU. Upgrades the limitation from a hedge to a measurement.
- **k-hop oracle connectivity** — are gold documents reachable within 1/2/3 hops, or at least in one component? Current `oracle_connectivity` is **1-hop only** (`g.has_edge`). Separates *graph failure* (no path exists) from *retrieval failure* (path exists, algorithm missed it).
- **Per-query error analysis** — queries where one graph type succeeds and another fails; explains which question types favour which construction.

---

## 12. Known Limitations

1. The comparison is not topology-only; edge attributes are part of the representation PCST consumes.
2. Metadata and semantic edge templates are title-based while entity edges use entity names.
3. The combined graph is a union and is therefore denser than any component; it is never density-matched.
4. `semantic_pcst` returns the smallest subgraphs, so its precision/F1 advantage may partly reflect retrieved size.
5. Thresholds were calibrated, then fixed — not exhaustively optimised.
6. `topk_e` was not varied in the PCST sensitivity analysis.
7. Retrieval budget K varies across regimes (11–15), confounding cross-regime recall comparisons. Within-regime comparisons are unaffected; the effect is bounded at ≈0.008 of delta per document.
8. Mutual k-NN retains only ~50% of k as realized degree, so k is not the achieved average degree.
9. Random baselines are required before claiming real structure beats generic structure.
10. Template-controlled ablations (titles added to entity edges, or removed from semantic/metadata) are future work.
11. Random baselines use a single seed (123); variance across seeds is unmeasured. The seed sweep in §10 shows shuffled-graph oracle connectivity ranging 0.0034–0.0291 around a density of 0.0168, so single-draw variance is not negligible.
12. **No confidence intervals.** Matched-size precision and F1 differences are 0.0004–0.002; these are reported as "collapsed / indistinguishable", never as an advantage. A query-level bootstrap would be needed to claim any of them.
13. Matched-size values in §8 are **linear interpolations** between sweep points, valid only within the 8.68–11.89 overlap band. The measured near-matched pairs are preferred evidence.
14. The recall↔oracle correspondence rests on four graph types in a single regime (~4% chance of a coincidental rank match) and cannot be tested at k=10 or k=25, which have only one PCST setting each.
15. 168 of 2255 evaluable queries cite the same document more than once. Handled by deduping (§6), but it means gold-set sizes are smaller than the raw evidence lists suggest.

---

## 13. Recommended Claim Language

**Use:**
- "In the completed default-PCST runs across the tested regimes, semantic graph construction yielded the most precise and compact retrieval, at the cost of recall."
- "We compare practical graph representations induced by different graph construction strategies."
- "Thresholds were selected through preliminary calibration and then fixed for the final controlled comparisons."

**Avoid:**
- "Semantic topology is better than entity topology."
- "The experiment isolates the pure effect of edge type."
- "The thresholds were optimised."
- "k = 17 is density-matched to the baseline entity graph." *(it is not — retention is ~50%)*

**Not yet — pending the remaining experiments (see §7, "Not yet established"):**
- "Semantic is the best graph construction strategy overall."
- "Real graph structure outperforms random graph structure."
- "The advantage is independent of retrieved size."

---

## 14. Artifacts: what is canonical, what is not

**Canonical results** — post-gold-dedup, in `results/<arm>/metrics/`:
`summary_table.csv`, `by_qtype_table.csv`, `graph_stats.csv`.
The pre-correction originals are archived as `results/<arm>/metrics_pre_gold_dedup/` and must not be used.

Corrected analysis: `results/analysis/pcst_sensitivity_k17_gold_dedup.csv`, `pcst_sensitivity_interpolated_by_size_gold_dedup.csv`.

**DO NOT USE** — legacy, flat-layout, produced under a *different config* (`pcst.topk = 3`, different metadata fields), predating the experiment runner:
`results/metrics/`, `results/retrieval/`, `results2/`. Rename to `_legacy_pre_runner` or treat as historical only. They are still tracked in git and will appear in a fresh clone.

**Repo vs Drive.** Committed: code, corrected metric and analysis CSVs, per-arm `config.yaml` and `manifest.json` (small, and the real provenance record — they carry the git SHA and the exact merged config each run consumed), documentation. Drive only: `data/processed/`, `data/graphs/` (~600 MB; edge embeddings dominate at 4 KB per directed edge), `results/*/retrieval/*.jsonl` (~6–7 MB per arm), and logs. Per-query data is required for error analysis and any bootstrap, so the repo is **not** self-sufficient for those.

---

## 15. Current Status

**Completed:** baseline/null (diagnostic), `mutual_knn_k10/17/25`; consolidated cross-regime comparison; PCST sensitivity on k=17 (5 settings × 4 graph types) with matched-size interpolation; empty-gold exclusion; gold-dedup correction of recall/F1/oracle across all 8 arms; Option C fix to shuffled-nodes with in-memory validation; reproduction check (baseline reproduces the earlier `results2` run on 7 of 9 conditions, graph stats identical).

**Next:** random graph baselines — k=17, default PCST, seed 123, both nulls, both PCST and no-PCST, new arm `k17_nulls`, with the three wiring edits in §10.

**Then, optional and CPU-only:** edge-prize distributions, k-hop oracle connectivity, per-query error analysis, bootstrap intervals.