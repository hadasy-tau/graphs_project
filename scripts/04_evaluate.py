"""Phase 4: Compute metrics for all retrieval conditions and save summary tables."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RETRIEVAL = ROOT / "results" / "retrieval"
METRICS = ROOT / "results" / "metrics"


def main():
    from src.evaluation.evaluator import evaluate_all

    evaluate_all(RETRIEVAL, METRICS)

    # Pretty-print the summary table
    import csv
    summary = METRICS / "summary_table.csv"
    with open(summary) as f:
        rows = list(csv.DictReader(f))

    print("\n=== Retrieval Results Summary ===")
    header = f"{'Condition':<30} {'Recall':>8} {'HitRate':>8} {'Prec':>8} {'F1':>8} {'Size':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['condition']:<30} "
            f"{row['evidence_recall']:>8} "
            f"{row['full_hit_rate']:>8} "
            f"{row['precision']:>8} "
            f"{row['f1']:>8} "
            f"{row['avg_retrieved_size']:>6}"
        )

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
