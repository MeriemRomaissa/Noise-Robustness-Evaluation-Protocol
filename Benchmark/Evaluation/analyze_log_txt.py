"""Read LaBraM-style `log.txt` files and summarize benchmark seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_log_txt(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL epoch log."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_seed(seed_dir: Path, selection_metric: str) -> dict[str, Any]:
    """Summarize one `seed_<n>` folder from its epoch log."""
    log_path = seed_dir / "log.txt"
    if not log_path.is_file():
        return {"seed_dir": str(seed_dir), "status": "MISSING_LOG"}

    records = parse_log_txt(log_path)
    if not records:
        return {"seed_dir": str(seed_dir), "status": "EMPTY_LOG"}

    best_record = max(records, key=lambda item: float(item[selection_metric]))
    last_record = records[-1]
    return {
        "seed_dir": str(seed_dir),
        "status": "PASS",
        "epochs_logged": len(records),
        "best_epoch": int(best_record["epoch"]),
        "selection_metric": selection_metric,
        "best_validation_value": float(best_record[selection_metric]),
        "last_epoch": int(last_record["epoch"]),
        "test_accuracy_best": best_record.get("test_accuracy"),
        "test_balanced_accuracy_best": best_record.get("test_balanced_accuracy"),
        "test_auroc_best": best_record.get("test_auroc"),
        "test_auprc_best": best_record.get("test_auprc"),
        "test_accuracy_last": last_record.get("test_accuracy"),
        "test_balanced_accuracy_last": last_record.get("test_balanced_accuracy"),
        "n_parameters": best_record.get("n_parameters"),
        "log_path": str(log_path),
    }


def summarize_output_root(output_root: Path, selection_metric: str) -> list[dict[str, Any]]:
    """Summarize every `seed_*` folder under one model output root."""
    seed_dirs = sorted(path for path in output_root.glob("seed_*") if path.is_dir())
    return [summarize_seed(path, selection_metric) for path in seed_dirs]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-metric", default="val_balanced_accuracy")
    parser.add_argument("--summary-json")
    parser.add_argument("--summary-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_root = Path(args.output_root).expanduser()
    rows = summarize_output_root(output_root, args.selection_metric)
    print(json.dumps(rows, indent=2, default=str))

    if args.summary_json:
        write_json(Path(args.summary_json).expanduser(), rows)
    if args.summary_csv:
        write_csv(Path(args.summary_csv).expanduser(), rows)


if __name__ == "__main__":
    main()
