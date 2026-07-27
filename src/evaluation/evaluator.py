"""Load retrieval results and compute aggregated metrics."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.data.loader import load_jsonl
from src.evaluation.metrics import aggregate, aggregate_by_qtype

logger = logging.getLogger(__name__)


def evaluate_condition(results_path: str | Path) -> dict:
    results = load_jsonl(results_path)
    metrics = aggregate(results)
    metrics["condition"] = Path(results_path).stem
    return metrics


def evaluate_all(results_dir: str | Path, output_dir: str | Path) -> None:
    """Evaluate every JSONL file in results_dir and save summary CSVs."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    qtype_rows = []

    for path in sorted(results_dir.glob("*.jsonl")):
        results = load_jsonl(path)
        metrics = aggregate(results)
        metrics["condition"] = path.stem
        rows.append(metrics)
        logger.info("[%s] recall=%.4f hit_rate=%.4f f1=%.4f size=%.1f",
                    path.stem, metrics["evidence_recall"], metrics["full_hit_rate"],
                    metrics["f1"], metrics["avg_retrieved_size"])

        by_qtype = aggregate_by_qtype(results)
        for qtype, qmetrics in by_qtype.items():
            qtype_rows.append({"condition": path.stem, "question_type": qtype, **qmetrics})

    _write_csv(rows, output_dir / "summary_table.csv")
    _write_csv(qtype_rows, output_dir / "by_qtype_table.csv")
    logger.info("Saved summary tables to %s", output_dir)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
