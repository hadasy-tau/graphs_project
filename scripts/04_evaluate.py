"""Phase 4: Compute metrics for all retrieval conditions and save summary tables."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from _retrieval_common import RETRIEVAL, METRICS


def main():
    from src.evaluation.evaluator import evaluate_all

    evaluate_all(RETRIEVAL, METRICS)

    # Pretty-print the summary table
    import csv
    summary = METRICS / "summary_table.csv"
    with open(summary) as f:
        rows = list(csv.DictReader(f))

    print("\n=== Retrieval Results Summary ===")
    header = (
        f"{'Condition':<30} {'Recall':>8} {'HitRate':>8} {'Prec':>8} "
        f"{'F1':>8} {'Size':>6} {'NEval':>7} {'Excl':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['condition']:<30} "
            f"{row['evidence_recall']:>8} "
            f"{row['full_hit_rate']:>8} "
            f"{row['precision']:>8} "
            f"{row['f1']:>8} "
            f"{row['avg_retrieved_size']:>6} "
            f"{row['n_queries']:>7} "
            f"{row['n_excluded_empty_gold']:>6}"
        )

    if rows:
        raw = {int(r["n_raw_queries"]) for r in rows}
        excl = {int(r["n_excluded_empty_gold"]) for r in rows}
        if len(raw) == 1 and len(excl) == 1:
            print(
                f"\nNEval = queries scored (evaluable). "
                f"Excl = queries excluded for having no gold evidence "
                f"({excl.pop()} of {raw.pop()} raw). "
                f"See the log above for the empty_gold_by_qtype breakdown."
            )
        else:
            print(
                "\nNEval = queries scored (evaluable). "
                "Excl = queries excluded for having no gold evidence. "
                "Counts differ across conditions - see the per-condition rows in "
                "summary_table.csv and the warning in the log above."
            )

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
