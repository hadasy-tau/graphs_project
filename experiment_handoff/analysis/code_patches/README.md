# Code patch artifacts

The previous pre-random handoff included a large generated patch file:

`local_fixes_gold_dedup_and_option_c.patch`

That file was intentionally omitted from the consolidated handoff because it was about 26 MB and duplicated changes that are now represented directly in the repository history.

Relevant committed fixes:
- duplicate gold-document handling in evaluation
- oracle connectivity gold deduplication
- shuffled-nodes Option C indexing fix

The current consolidated handoff keeps the experimental decisions, corrected metrics, PCST sensitivity tables, and post-random baseline results.
