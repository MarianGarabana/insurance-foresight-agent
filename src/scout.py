"""
scout.py
--------
STEP 1 — Candidate Discovery

The Scout takes a natural language query and returns a list of candidate
venture names that might be relevant to the research question.

v0.2 — Search-grounded discovery
---------------------------------
The Scout now runs in two modes depending on configuration:

MODE A: Search-grounded (PREFERRED — requires TAVILY_API_KEY)
  1. Call search_web() to get real web results for the query.
  2. Pass those results to the LLM as context.
  3. LLM extracts company names FROM the search results (parser, not generator).
  4. Each candidate includes source_urls pointing to the search results that
     mentioned it — providing an evidence trail.

MODE B: LLM-only (FALLBACK — used when no search API is configured)
  1. Ask the LLM to generate company names from its training knowledge.
  2. LLM is the sole source — more risk of hallucination and stale data.
  This is identical to v0.1 behavior.

The mode is selected automatically based on settings.search_enabled.
A log message tells you which mode was used on each run.
"""

from src.config import settings
from src.llm_client import call_llm
from src.logger import get_logger
from src.models import CandidateVenture
from src.prompts import (
    SCOUT_SYSTEM_PROMPT,
    SCOUT_FROM_SEARCH_SYSTEM_PROMPT,
    build_scout_prompt,
    build_scout_from_search_prompt,
)
from src.search import search_web, format_results_for_prompt
from src.utils import parse_json_response, safe_get

log = get_logger(__name__)


def generate_candidates(query: str, max_candidates: int | None = None) -> list[CandidateVenture]:
    """
    Generate a list of candidate ventures for a given research query.

    Automatically selects search-grounded or LLM-only mode based on whether
    a search API key is configured. Search-grounded mode is strongly preferred.

    Args:
        query: The researcher's natural language query.
               Example: "Find flood-risk ventures in Europe relevant to insurance"
        max_candidates: How many candidates to request. Defaults to settings.max_candidates.

    Returns:
        A list of CandidateVenture objects. May be empty if the LLM returns nothing valid.
    """
    if max_candidates is None:
        max_candidates = settings.max_candidates

    log.info(f"Scout: generating up to {max_candidates} candidates for query: '{query}'")

    if settings.search_enabled:
        log.info("Scout: running in SEARCH-GROUNDED mode (Tavily search active).")
        return _generate_from_search(query=query, max_candidates=max_candidates)
    else:
        log.info(
            "Scout: running in LLM-ONLY mode (no search API configured). "
            "Set TAVILY_API_KEY in .env for better results."
        )
        return _generate_llm_only(query=query, max_candidates=max_candidates)


def _generate_from_search(query: str, max_candidates: int) -> list[CandidateVenture]:
    """
    Search-grounded candidate generation (Mode A).

    1. Fetch web search results for the query.
    2. Pass them to the LLM as context.
    3. LLM extracts real company names from the results.

    If search returns zero results (API error, bad key, etc.), falls back
    to LLM-only mode automatically.

    Args:
        query: The research query.
        max_candidates: Maximum number of candidates to extract.

    Returns:
        A list of CandidateVenture objects with source_urls populated.
    """
    # Step 1: Get real search results
    search_results = search_web(query=query, max_results=max_candidates + 5)
    # We request a few extra results to give the LLM more material to work with.

    if not search_results:
        log.warning(
            "Scout: search returned no results. "
            "Falling back to LLM-only mode for this query."
        )
        return _generate_llm_only(query=query, max_candidates=max_candidates)

    # Step 2: Format results for the LLM prompt
    formatted = format_results_for_prompt(search_results)

    # Step 3: Ask LLM to extract companies from the search results
    user_prompt = build_scout_from_search_prompt(
        query=query,
        max_candidates=max_candidates,
        formatted_search_results=formatted,
    )

    raw_response = call_llm(
        system_prompt=SCOUT_FROM_SEARCH_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context="scout-search",
        temperature=0.1,  # Low: we want precise extraction, not creativity
    )

    if raw_response is None:
        log.error("Scout: LLM extraction from search results failed. Returning empty list.")
        return []

    return _parse_candidates(raw_response, context="scout-search")


def _generate_llm_only(query: str, max_candidates: int) -> list[CandidateVenture]:
    """
    LLM-only candidate generation (Mode B — v0.1 fallback).

    Ask the LLM to generate company names from its training knowledge only.
    No search grounding — higher risk of hallucination and stale data.

    Args:
        query: The research query.
        max_candidates: Maximum number of candidates to generate.

    Returns:
        A list of CandidateVenture objects with empty source_urls.
    """
    user_prompt = build_scout_prompt(query=query, max_candidates=max_candidates)

    raw_response = call_llm(
        system_prompt=SCOUT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context="scout-llm",
        temperature=0.2,  # Slightly higher: allow the LLM to recall diverse companies
    )

    if raw_response is None:
        log.error("Scout: LLM returned no response. Returning empty candidate list.")
        return []

    return _parse_candidates(raw_response, context="scout-llm")


def _parse_candidates(raw_response: str, context: str) -> list[CandidateVenture]:
    """
    Parse the LLM's JSON response into a list of CandidateVenture objects.

    Both search-grounded and LLM-only modes return the same JSON format:
    {"candidates": [{"name", "suggested_website", "source_urls", "reasoning"}, ...]}

    Args:
        raw_response: Raw JSON string from the LLM.
        context: Label for log messages.

    Returns:
        A list of CandidateVenture objects. Empty list if parsing fails.
    """
    parsed = parse_json_response(raw_response, context=context)
    if parsed is None:
        log.error(f"[{context}] Could not parse LLM response as JSON. Returning empty list.")
        return []

    raw_candidates = parsed.get("candidates", [])
    if not raw_candidates:
        log.warning(f"[{context}] LLM returned zero candidates.")
        return []

    candidates: list[CandidateVenture] = []
    for item in raw_candidates:
        try:
            # source_urls may be missing in LLM-only mode — default to empty list
            source_urls = item.get("source_urls", [])
            if not isinstance(source_urls, list):
                source_urls = []

            candidate = CandidateVenture(
                name=safe_get(item, "name", default="Unknown"),
                suggested_website=item.get("suggested_website") or None,
                scout_reasoning=item.get("reasoning"),
                source_urls=source_urls,
            )
            candidates.append(candidate)
            log.debug(
                f"[{context}] Candidate: {candidate.name} "
                f"({candidate.suggested_website}) "
                f"sources={len(candidate.source_urls)}"
            )
        except Exception as e:
            log.warning(f"[{context}] Could not parse candidate item {item}: {e}")
            continue

    log.info(f"Scout: generated {len(candidates)} candidates.")
    return candidates
