"""
Aggregate batch results into a CSV.

For each logs_* directory found under the given batch root, reads:
  - summary.json  → all scalar fields
  - console.log   → the timing JSON block printed near the end

Rows = one per experiment run (identified by path relative to batch root).
Columns = all summary fields + timing fields (prefixed with "timing_").

Usage:
    python batch_to_csv.py <batch_dir> [--out results.csv]

Example:
    python batch_to_csv.py logging/batch_results_tpm --out results.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


_TIMING_HEADER = "Average Timing Profiles over tasks (in seconds):"


def _extract_timing(console_log: Path) -> dict:
    """Return the timing dict from the bottom of a console.log file, or {}."""
    try:
        text = console_log.read_text(errors="replace")
    except OSError:
        return {}

    idx = text.rfind(_TIMING_HEADER)
    if idx == -1:
        return {}

    # Find the JSON object that follows the header line
    after = text[idx + len(_TIMING_HEADER):]
    m = re.search(r"\{[^}]+\}", after, re.DOTALL)
    if not m:
        return {}

    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def _load_summary(summary_json: Path) -> dict:
    try:
        return json.loads(summary_json.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect_rows(batch_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(batch_dir.rglob("summary.json")):
        logs_dir = summary_path.parent
        # Row identifier: path from batch_dir to the logs_* folder, dropping the
        # logs_TIMESTAMP leaf so we get the experiment/model path.
        rel_parts = logs_dir.relative_to(batch_dir).parts
        if len(rel_parts) >= 2 and rel_parts[-1].startswith("logs_"):
            experiment_path = "/".join(rel_parts[:-1])
            timestamp = rel_parts[-1]
        else:
            experiment_path = "/".join(rel_parts)
            timestamp = ""

        summary = _load_summary(summary_path)
        timing = _extract_timing(logs_dir / "console.log")

        row = {
            "experiment": experiment_path,
            "timestamp": timestamp,
            **summary,
            **{f"timing_{k}": v for k, v in timing.items()},
        }
        rows.append(row)

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("No experiment results found.", file=sys.stderr)
        return

    # Union of all keys, preserving insertion order
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", help="Path to batch results directory")
    parser.add_argument("--out", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.exists():
        print(f"Error: {batch_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(batch_dir)
    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
