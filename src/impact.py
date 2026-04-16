"""
impact.py
---------
STEP 5 — Impact Analysis

For each verified venture, the Impact Analyser estimates how it affects
insurance outcomes across five areas (I1–I5):

  I1 — Premium Revenue
  I2 — Investment Revenue
  I3 — Claims Costs         ← most commonly affected; I3 reduction = fewer/smaller claims
  I4 — Operational Costs
  I5 — Capital & Reserves

The analysis calls the LLM with a structured prompt and parses the result
into an ImpactResult model.

Added in v0.3.
"""

from typing import Optional

from src.llm_client import call_llm
from src.logger import get_logger
from src.models import ImpactResult, VentureRecord, VALID_IMPACT_AREAS, VALID_IMPACT_DIRECTIONS, VALID_TIME_HORIZONS
from src.prompts import IMPACT_SYSTEM_PROMPT, build_impact_prompt
from src.utils import parse_json_response

log = get_logger(__name__)


def analyze_impact(venture: VentureRecord) -> Optional[ImpactResult]:
    """
    Estimate the insurance impact of a verified venture.

    Calls the LLM using the venture's description, driver tags, and category
    to produce a structured ImpactResult.

    Args:
        venture: A fully verified and classified VentureRecord.

    Returns:
        An ImpactResult if the analysis succeeded, or None on failure.
    """
    log.info(f"Impact: analysing '{venture.venture_name}'")

    user_prompt = build_impact_prompt(
        venture_name=venture.venture_name,
        short_description=venture.short_description,
        driver_tags=venture.driver_tags,
        category=venture.category,
        source_urls=venture.source_urls,
    )

    raw_response = call_llm(
        system_prompt=IMPACT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context="impact",
        temperature=0.1,
    )

    if raw_response is None:
        log.warning(f"Impact: LLM call failed for '{venture.venture_name}'. Skipping.")
        return None

    parsed = parse_json_response(raw_response, context="impact")
    if parsed is None:
        log.warning(f"Impact: JSON parse failed for '{venture.venture_name}'. Skipping.")
        return None

    # Validate and normalise impact_areas
    raw_areas = parsed.get("impact_areas", [])
    impact_areas = _validate_impact_areas(raw_areas, venture.venture_name)

    # Validate impact_direction
    raw_direction = parsed.get("impact_direction", "")
    impact_direction = _validate_impact_direction(raw_direction, venture.venture_name)

    # Validate time_horizon
    raw_horizon = parsed.get("time_horizon", "")
    time_horizon = _validate_time_horizon(raw_horizon, venture.venture_name)

    # Clamp confidence to 0.0–1.0
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (ValueError, TypeError):
        confidence = 0.5

    drivers = parsed.get("drivers", venture.driver_tags)
    if not isinstance(drivers, list):
        drivers = venture.driver_tags

    mechanism = parsed.get("mechanism", "")
    insurance_effect = parsed.get("insurance_effect", "")

    result = ImpactResult(
        drivers=drivers,
        impact_areas=impact_areas,
        impact_direction=impact_direction,
        mechanism=mechanism,
        insurance_effect=insurance_effect,
        time_horizon=time_horizon,
        confidence=confidence,
    )

    log.debug(
        f"Impact: '{venture.venture_name}' → {result.impact_areas} / "
        f"{result.impact_direction} / {result.time_horizon} "
        f"(confidence={result.confidence:.2f})"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_impact_areas(raw_areas: list, venture_name: str) -> list[str]:
    """Filter impact areas to only valid I1–I5 codes. Defaults to ['I3'] on failure."""
    if not isinstance(raw_areas, list):
        log.warning(f"Impact: impact_areas for '{venture_name}' is not a list. Defaulting to I3.")
        return ["I3"]

    valid = [a for a in raw_areas if isinstance(a, str) and a.upper() in VALID_IMPACT_AREAS]
    if not valid:
        log.warning(f"Impact: no valid impact areas for '{venture_name}'. Defaulting to I3.")
        return ["I3"]

    return [a.upper() for a in valid]


def _validate_impact_direction(raw: str, venture_name: str) -> str:
    """Normalise impact direction. Defaults to 'mixed' if invalid."""
    normalised = raw.strip().lower() if isinstance(raw, str) else ""
    if normalised in VALID_IMPACT_DIRECTIONS:
        return normalised

    log.warning(
        f"Impact: invalid impact_direction '{raw}' for '{venture_name}'. Defaulting to 'mixed'."
    )
    return "mixed"


def _validate_time_horizon(raw: str, venture_name: str) -> str:
    """Normalise time horizon. Defaults to 'medium' if invalid."""
    normalised = raw.strip().lower() if isinstance(raw, str) else ""
    if normalised in VALID_TIME_HORIZONS:
        return normalised

    log.warning(
        f"Impact: invalid time_horizon '{raw}' for '{venture_name}'. Defaulting to 'medium'."
    )
    return "medium"
