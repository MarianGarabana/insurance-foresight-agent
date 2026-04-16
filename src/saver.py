"""
saver.py
--------
STEP 5 — Save Results

The Saver writes the final results to disk in two formats:
- CSV (easy to open in Excel or Google Sheets)
- JSON (easy to process programmatically)

v0.2 changes:
- Timestamped filenames prevent overwriting previous runs.
  Example: verified_ventures_20260416_143022.csv
- source_urls column is included in the CSV for evidence tracking.
- A run_id (timestamp string) is generated once per run and shared across
  all three output files so they can be matched by filename.

Two separate files are written:
- output/verified_ventures_TIMESTAMP.csv / .json  → verified ventures
- output/rejected_candidates_TIMESTAMP.csv        → rejected ventures

TODO (future improvements):
- Add Notion API integration to push results to a Notion database.
- Add SQLite storage for querying across multiple runs.
- Add a "latest" symlink that always points to the most recent run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import settings
from src.logger import get_logger
from src.models import RejectedCandidate, VentureRecord

log = get_logger(__name__)


def make_run_id() -> str:
    """
    Generate a timestamp string to use as a run identifier.

    Called once per run in main.py and passed to all save functions so that
    the three output files (verified CSV, verified JSON, rejected CSV) all share
    the same timestamp and can be easily matched.

    Returns:
        A timestamp string in the format YYYYMMDD_HHMMSS, e.g. "20260416_143022".
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_verified_ventures(ventures: list[VentureRecord], run_id: str) -> Path:
    """
    Save all verified and classified ventures to CSV and JSON.

    Files written:
    - output/verified_ventures_{run_id}.csv
    - output/verified_ventures_{run_id}.json

    Args:
        ventures: List of VentureRecord objects to save.
        run_id: Timestamp string from make_run_id(). Used in filenames.

    Returns:
        Path to the CSV file that was written (useful for printing to user).
    """
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"verified_ventures_{run_id}.csv"
    json_path = output_dir / f"verified_ventures_{run_id}.json"

    if not ventures:
        log.warning("Saver: no verified ventures to save.")
        # Write empty files with headers so they always exist
        pd.DataFrame(columns=_CSV_COLUMN_ORDER).to_csv(csv_path, index=False)
        _write_json(ventures=[], json_path=json_path, run_id=run_id)
        return csv_path

    # ── CSV output ────────────────────────────────────────────────────── #
    rows = [v.to_flat_dict() for v in ventures]
    df = pd.DataFrame(rows)

    # Reorder columns for readability in Excel / Sheets
    ordered_columns = [c for c in _CSV_COLUMN_ORDER if c in df.columns]
    df = df[ordered_columns]

    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info(f"Saver: saved {len(ventures)} verified ventures to {csv_path}")

    # ── JSON output ───────────────────────────────────────────────────── #
    _write_json(ventures=ventures, json_path=json_path, run_id=run_id)

    return csv_path


def save_rejected_candidates(rejected: list[RejectedCandidate], run_id: str) -> Path:
    """
    Save all rejected candidates to a CSV file.

    File written:
    - output/rejected_candidates_{run_id}.csv

    Args:
        rejected: List of RejectedCandidate objects to save.
        run_id: Timestamp string from make_run_id(). Used in filenames.

    Returns:
        Path to the CSV file that was written.
    """
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"rejected_candidates_{run_id}.csv"

    rejected_columns = ["candidate_name", "suggested_website", "rejection_reason", "http_status"]

    if not rejected:
        log.info("Saver: no rejected candidates to save.")
        pd.DataFrame(columns=rejected_columns).to_csv(csv_path, index=False)
        return csv_path

    rows = [r.model_dump() for r in rejected]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    log.info(f"Saver: saved {len(rejected)} rejected candidates to {csv_path}")
    return csv_path


def print_summary(
    verified: list[VentureRecord],
    rejected: list[RejectedCandidate],
    run_id: str,
) -> None:
    """
    Print a plain-text summary of the run to the console.

    Args:
        verified: List of verified VentureRecord objects.
        rejected: List of RejectedCandidate objects.
        run_id: Timestamp string used to build output filenames.
    """
    total = len(verified) + len(rejected)
    print()
    print("=" * 62)
    print("VENTURE SCOUT — RUN SUMMARY")
    print("=" * 62)
    print(f"  Candidates evaluated : {total}")
    print(f"  Verified             : {len(verified)}")
    print(f"  Rejected             : {len(rejected)}")

    if verified:
        print()
        print("  Verified ventures:")
        for v in verified:
            tags = ", ".join(v.driver_tags)
            confidence_pct = int(v.confidence * 100)
            sources = f"  {len(v.source_urls)} source(s)" if v.source_urls else ""
            print(
                f"    • {v.venture_name:<33} {v.category:<10} "
                f"[{tags}]  {confidence_pct}%{sources}"
            )

    if rejected:
        print()
        print("  Rejected candidates:")
        for r in rejected:
            reason = (
                r.rejection_reason[:58] + "..."
                if len(r.rejection_reason) > 58
                else r.rejection_reason
            )
            print(f"    ✗ {r.candidate_name:<33} {reason}")

    print()
    print(f"  Output directory : {settings.output_dir}/")
    print(f"  Run ID           : {run_id}")
    print(f"  Verified CSV     : verified_ventures_{run_id}.csv")
    print(f"  Rejected CSV     : rejected_candidates_{run_id}.csv")
    print(f"  Log file         : logs/run.log")
    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Column order for the verified ventures CSV.
# Columns are shown in this order when opened in Excel / Google Sheets.
_CSV_COLUMN_ORDER = [
    "venture_name",
    "website",
    "location",
    "stage",
    "total_funding",
    "insurance_sector",
    "category",
    "driver_tags",
    "confidence",
    "short_description",
    "classification_reasoning",
    "source_urls",            # v0.2: evidence trail
    "extraction_confidence",  # v0.2: how confident the extractor was
    # v0.3: impact analysis columns
    "impact_areas",           # pipe-separated I1–I5 codes, e.g. "I3|I5"
    "impact_direction",       # "positive", "negative", or "mixed"
    "mechanism",              # short technical explanation
    "time_horizon",           # "short", "medium", or "long"
    "impact_confidence",      # 0.0–1.0
]


def _write_json(
    ventures: list[VentureRecord],
    json_path: Path,
    run_id: str,
) -> None:
    """
    Write verified ventures to a JSON file with metadata.

    Args:
        ventures: List of VentureRecord objects.
        json_path: Where to write the file.
        run_id: Run timestamp for metadata.
    """
    output_data = {
        "metadata": {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_verified": len(ventures),
        },
        "ventures": [v.model_dump() for v in ventures],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log.info(f"Saver: saved {len(ventures)} verified ventures to {json_path}")
