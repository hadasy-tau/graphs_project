"""Load retrieval results and compute aggregated metrics."""
from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

from src.data.loader import load_jsonl
from src.evaluation.metrics import aggregate, aggregate_by_qtype

logger = logging.getLogger(__name__)

# The nine retrieval conditions produced by scripts/03_*. evaluate_all refuses to
# build a summary unless all are present, and ignores any other .jsonl file, so
# a partial run or a stale file from an earlier experiment can't leak in silently.
EXPECTED_CONDITIONS = {
    "entity_pcst",
    "entity_no_pcst",
    "metadata_pcst",
    "metadata_no_pcst",
    "semantic_pcst",
    "semantic_no_pcst",
    "combined_pcst",
    "combined_no_pcst",
    "dense_no_pcst",
}


def split_evaluable_results(results: list[dict]) -> tuple[list[dict], dict]:
    """Split retrieval results into evaluable ones plus filtering diagnostics.

    A query is evaluable only if it has at least one gold document. Queries with
    an empty gold set - MultiHop-RAG null queries, plus any evidence titles the
    loader could not resolve - would otherwise score evidence_recall = 1.0 and
    full_hit_rate = True by vacuous truth. That hands every condition the same
    block of free perfect scores, inflating absolute numbers and compressing the
    differences between conditions, which is exactly the signal we measure.

    Returns:
        (evaluable_results, diagnostics) where diagnostics contains
        n_raw_queries, n_evaluable_queries, n_excluded_empty_gold and
        empty_gold_by_qtype.
    """
    evaluable = [r for r in results if r.get("gold_doc_ids")]
    excluded = [r for r in results if not r.get("gold_doc_ids")]

    diagnostics = {
        "n_raw_queries": len(results),
        "n_evaluable_queries": len(evaluable),
        "n_excluded_empty_gold": len(excluded),
        "empty_gold_by_qtype": dict(
            Counter(r.get("question_type", "unknown") for r in excluded)
        ),
    }
    return evaluable, diagnostics


def evaluate_all(results_dir: str | Path, output_dir: str | Path) -> None:
    """Evaluate the expected retrieval conditions and save summary CSVs.

    Only the conditions in EXPECTED_CONDITIONS are evaluated; any other .jsonl
    file in results_dir is ignored with a warning. Raises if an expected
    condition file is missing, or if a condition has no evaluable queries left
    after empty-gold filtering.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(results_dir.glob("*.jsonl"))
    found = {p.stem for p in paths}
    _require_all_conditions(results_dir, found)
    _warn_unexpected_files(results_dir, found)
    paths = [p for p in paths if p.stem in EXPECTED_CONDITIONS]

    rows: list[dict] = []
    qtype_rows: list[dict] = []
    excluded_counts: dict[str, int] = {}

    for path in paths:
        condition = path.stem
        results = load_jsonl(path)
        evaluable, diag = split_evaluable_results(results)
        _log_diagnostics(condition, diag)
        _require_evaluable(condition, evaluable, diag)

        excluded_counts[condition] = diag["n_excluded_empty_gold"]

        row = _build_row(condition, evaluable, diag)
        rows.append(row)
        logger.info(
            "[%s] recall=%.4f hit_rate=%.4f f1=%.4f size=%.1f n=%d",
            condition, row["evidence_recall"], row["full_hit_rate"],
            row["f1"], row["avg_retrieved_size"], row["n_queries"],
        )

        for qtype, qmetrics in aggregate_by_qtype(evaluable).items():
            qtype_rows.append(
                {"condition": condition, "question_type": qtype, **qmetrics}
            )

    _warn_if_inconsistent(excluded_counts)

    _write_csv(rows, output_dir / "summary_table.csv")
    _write_csv(qtype_rows, output_dir / "by_qtype_table.csv")
    logger.info("Saved summary tables to %s", output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_row(condition: str, evaluable: list[dict], diag: dict) -> dict:
    """Aggregate metrics for one condition and attach filtering diagnostics.

    n_queries keeps its existing meaning - the number of queries actually
    scored, i.e. the evaluable count.
    """
    metrics = aggregate(evaluable)
    return {
        "condition": condition,
        **metrics,
        "n_raw_queries": diag["n_raw_queries"],
        "n_excluded_empty_gold": diag["n_excluded_empty_gold"],
    }


def _require_all_conditions(results_dir: Path, found: set[str]) -> None:
    missing = EXPECTED_CONDITIONS - found
    if missing:
        raise FileNotFoundError(
            f"Missing retrieval result files for condition(s): "
            f"{', '.join(sorted(missing))}. Expected all "
            f"{len(EXPECTED_CONDITIONS)} conditions in {results_dir}. "
            f"Run the corresponding scripts/03_*.py script(s) before evaluating."
        )


def _warn_unexpected_files(results_dir: Path, found: set[str]) -> None:
    """Stale result files from earlier experiments must not enter the summary."""
    unexpected = found - EXPECTED_CONDITIONS
    if unexpected:
        logger.warning(
            "Ignoring %d unexpected .jsonl file(s) in %s: %s. Only the %d expected "
            "conditions are evaluated.",
            len(unexpected), results_dir, sorted(unexpected), len(EXPECTED_CONDITIONS),
        )


def _require_evaluable(condition: str, evaluable: list[dict], diag: dict) -> None:
    if not evaluable:
        raise ValueError(
            f"No evaluable queries remain for condition {condition} after "
            f"filtering empty-gold queries "
            f"({diag['n_raw_queries']} raw queries, all with empty gold_doc_ids)."
        )


def _log_diagnostics(condition: str, diag: dict) -> None:
    if diag["n_excluded_empty_gold"]:
        logger.info(
            "[%s] empty-gold filtering: %d raw -> %d evaluable, %d excluded. "
            "empty_gold_by_qtype=%s",
            condition,
            diag["n_raw_queries"],
            diag["n_evaluable_queries"],
            diag["n_excluded_empty_gold"],
            diag["empty_gold_by_qtype"],
        )
    else:
        logger.info(
            "[%s] empty-gold filtering: all %d queries have gold evidence; none excluded.",
            condition, diag["n_raw_queries"],
        )


def _warn_if_inconsistent(excluded_counts: dict[str, int]) -> None:
    """All conditions share one query set, so the excluded count should match."""
    if len(set(excluded_counts.values())) > 1:
        logger.warning(
            "Empty-gold count differs across conditions: %s. All conditions should "
            "be run over the same query set, so this suggests some result files are "
            "stale or were produced from a different queries.jsonl.",
            excluded_counts,
        )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
