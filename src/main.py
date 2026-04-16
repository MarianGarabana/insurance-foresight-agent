"""
main.py
-------
CLI entry point for the Insurance Foresight Agent.

Run it like this:
    python -m src.main scout "Find flood-risk ventures in Europe relevant to insurance"

How it works:
1. Parse the CLI arguments.
2. Run the full Scout workflow: generate → verify → extract → classify → save.
3. Print a summary to the terminal.

This file is intentionally kept short. The actual logic lives in the
other modules (scout.py, verifier.py, extractor.py, classifier.py, saver.py).
main.py just wires them together in the right order.
"""

import argparse
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import settings
from src.logger import get_logger
from src.models import RejectedCandidate, VentureRecord
from src.scout import generate_candidates
from src.verifier import verify_venture
from src.extractor import extract_fields
from src.classifier import classify_venture
from src.saver import save_verified_ventures, save_rejected_candidates, print_summary, make_run_id

log = get_logger(__name__)
console = Console()


def run_scout_workflow(query: str, max_candidates: int) -> None:
    """
    Run the full Venture Scout pipeline for a given query.

    Pipeline steps:
        1. Scout     — generate candidate venture names from the query
        2. Verifier  — confirm each candidate has a live official website
        3. Extractor — extract structured fields (location, stage, funding, ...)
        4. Classifier — assign D1–D11 tags and Direct/Indirect category
        5. Saver     — write verified ventures and rejected candidates to disk

    Args:
        query: Natural language research query.
        max_candidates: How many candidates to ask the Scout to generate.
    """
    # Validate that an API key is available before making any LLM calls
    settings.validate_api_key()

    # Ensure output and log directories exist before we start
    settings.ensure_dirs()

    console.print()
    # Generate a run ID (timestamp) shared by all output files for this run
    run_id = make_run_id()

    search_mode = "search-grounded (Tavily)" if settings.search_enabled else "LLM-only"

    console.print(f"[bold cyan]Insurance Foresight — Venture Scout[/bold cyan]")
    console.print(f"[dim]Query:[/dim] {query}")
    console.print(f"[dim]Model:[/dim] {settings.openai_model}")
    console.print(f"[dim]Scout mode:[/dim] {search_mode}")
    console.print(f"[dim]Max candidates:[/dim] {max_candidates}")
    console.print(f"[dim]Run ID:[/dim] {run_id}")
    console.print()

    # ------------------------------------------------------------------ #
    # STEP 1 — Candidate Discovery
    # ------------------------------------------------------------------ #
    console.print("[bold]Step 1/4[/bold] — Generating candidates...")
    log.info(f"Starting scout workflow. Query: '{query}'")

    candidates = generate_candidates(query=query, max_candidates=max_candidates)

    if not candidates:
        console.print("[red]No candidates were generated. Check your query and API key.[/red]")
        log.error("Workflow aborted: no candidates generated.")
        sys.exit(1)

    console.print(f"[green]✓[/green] Generated {len(candidates)} candidate(s).")
    console.print()

    # ------------------------------------------------------------------ #
    # STEPS 2–4 — Verify, Extract, Classify each candidate
    # ------------------------------------------------------------------ #
    verified_ventures: list[VentureRecord] = []
    rejected_candidates: list[RejectedCandidate] = []

    console.print(f"[bold]Steps 2–4[/bold] — Verifying, extracting, and classifying...")
    console.print()

    for i, candidate in enumerate(candidates, start=1):
        console.print(f"  [{i}/{len(candidates)}] {candidate.name}")

        # STEP 2 — Verify
        log.info(f"Processing candidate {i}/{len(candidates)}: '{candidate.name}'")
        verification = verify_venture(candidate)

        if not verification.verified:
            # Rejected — record the reason and move on
            console.print(f"    [red]✗ Rejected:[/red] {verification.rejection_reason}")
            log.info(f"Rejected '{candidate.name}': {verification.rejection_reason}")

            rejected_candidates.append(
                RejectedCandidate(
                    candidate_name=candidate.name,
                    suggested_website=candidate.suggested_website,
                    rejection_reason=verification.rejection_reason,
                    http_status=verification.http_status,
                )
            )
            continue  # Skip to next candidate

        console.print(f"    [green]✓ Verified:[/green] {verification.website}")

        # STEP 3 — Extract fields
        # Small delay to avoid hammering the OpenAI API on every call
        time.sleep(0.5)
        fields = extract_fields(
            venture_name=verification.candidate_name,
            website=verification.website,
            source_urls=candidate.source_urls,  # Pass search evidence to extractor
        )
        console.print(
            f"    [dim]Extracted:[/dim] {fields.location} | {fields.stage} | {fields.total_funding}"
        )

        # STEP 4 — Classify
        time.sleep(0.5)
        classification = classify_venture(fields)
        tags_str = ", ".join(classification.driver_tags)
        confidence_pct = int(classification.confidence * 100)
        console.print(
            f"    [dim]Classified:[/dim] {classification.category} | [{tags_str}] | {confidence_pct}% confidence"
        )

        # Build the final VentureRecord
        venture_record = VentureRecord(
            # From extraction
            venture_name=fields.venture_name,
            website=fields.website,
            location=fields.location,
            short_description=fields.short_description,
            stage=fields.stage,
            total_funding=fields.total_funding,
            insurance_sector=fields.insurance_sector,
            # From classification
            driver_tags=classification.driver_tags,
            category=classification.category,
            confidence=classification.confidence,
            classification_reasoning=classification.reasoning,
            # Evidence trail (v0.2)
            source_urls=candidate.source_urls,
        )
        verified_ventures.append(venture_record)
        console.print()

    # ------------------------------------------------------------------ #
    # STEP 5 — Save Results
    # ------------------------------------------------------------------ #
    console.print("[bold]Step 5[/bold] — Saving results...")
    save_verified_ventures(verified_ventures, run_id=run_id)
    save_rejected_candidates(rejected_candidates, run_id=run_id)
    console.print("[green]✓[/green] Results saved.")

    # Print summary
    print_summary(verified_ventures, rejected_candidates, run_id=run_id)
    log.info(
        f"Workflow complete. Verified: {len(verified_ventures)}, "
        f"Rejected: {len(rejected_candidates)}"
    )


def main() -> None:
    """
    Parse CLI arguments and dispatch to the appropriate command.

    Currently supports one command: `scout`.

    Usage:
        python -m src.main scout "your query here"
        python -m src.main scout "your query here" --max-candidates 15
    """
    parser = argparse.ArgumentParser(
        prog="insurance-foresight",
        description="Insurance Industry Foresight Research System — Venture Scout Agent",
    )

    # Create subcommands (currently just "scout", but easy to add more)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------ #
    # "scout" subcommand
    # ------------------------------------------------------------------ #
    scout_parser = subparsers.add_parser(
        "scout",
        help="Run the Venture Scout workflow for a research query.",
    )
    scout_parser.add_argument(
        "query",
        type=str,
        help=(
            'Research query. Example: '
            '"Find flood-risk ventures in Europe relevant to insurance"'
        ),
    )
    scout_parser.add_argument(
        "--max-candidates",
        type=int,
        default=settings.max_candidates,
        help=f"Number of candidates to generate (default: {settings.max_candidates})",
    )
    scout_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory (default: from .env or 'output/')",
    )

    # Parse arguments
    args = parser.parse_args()

    # If no command was given, print help and exit
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Handle the scout command
    if args.command == "scout":
        # Allow output directory override from CLI
        if args.output_dir:
            from pathlib import Path
            settings.output_dir = Path(args.output_dir)

        run_scout_workflow(
            query=args.query,
            max_candidates=args.max_candidates,
        )

    # TODO: Add more commands here as the system grows, e.g.:
    # elif args.command == "monitor":
    #     run_monitor_workflow(...)
    # elif args.command == "report":
    #     run_report_workflow(...)


if __name__ == "__main__":
    main()
