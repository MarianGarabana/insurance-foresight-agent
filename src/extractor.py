"""
extractor.py
------------
STEP 3 — Field Extraction

The Extractor takes a verified venture (name + confirmed website) and
extracts structured fields using the LLM.

In v0.2, the extractor also accepts source_urls from the Scout so the LLM
has additional context from real search results when extracting fields.

If the LLM does not know the company well, fields will be "Unknown".
That is acceptable — accuracy is more important than completeness.

v0.2 changes:
- Uses shared llm_client.call_llm() instead of a private _call_llm()
- Accepts source_urls and passes them to the LLM as context
- Returns extraction_confidence to flag uncertain extractions

TODO (future improvements):
- Fetch and parse the company's website to extract real-time information.
- Use Crunchbase / PitchBook API to get accurate funding and stage data.
"""

from src.llm_client import call_llm
from src.logger import get_logger
from src.models import ExtractedFields
from src.prompts import EXTRACTOR_SYSTEM_PROMPT, build_extractor_prompt
from src.utils import parse_json_response, safe_get

log = get_logger(__name__)


def extract_fields(
    venture_name: str,
    website: str,
    source_urls: list[str] | None = None,
) -> ExtractedFields:
    """
    Extract structured fields for a verified venture using the LLM.

    If extraction fails for any reason, a fallback ExtractedFields object
    is returned with all optional fields set to "Unknown". We never crash
    the pipeline just because one extraction failed.

    Args:
        venture_name: The verified company name.
        website: The confirmed official website URL.
        source_urls: Optional list of web search URLs that mentioned this company.
                     Passed to the LLM as additional context to improve extraction.

    Returns:
        An ExtractedFields object with the best available information.
    """
    log.info(f"Extractor: extracting fields for '{venture_name}'")

    source_urls = source_urls or []

    # Build the user prompt, optionally including search context
    user_prompt = build_extractor_prompt(
        venture_name=venture_name,
        website=website,
        source_urls=source_urls,
    )

    # Call the LLM via shared client
    raw_response = call_llm(
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context="extractor",
        temperature=0.1,  # Very low — we want consistent, factual extraction
    )

    # If LLM call failed, return a minimal fallback object
    if raw_response is None:
        log.warning(f"Extractor: LLM failed for '{venture_name}'. Using fallback values.")
        return _fallback_fields(venture_name, website)

    # Parse the JSON response
    parsed = parse_json_response(raw_response, context="extractor")
    if parsed is None:
        log.warning(f"Extractor: JSON parse failed for '{venture_name}'. Using fallback values.")
        return _fallback_fields(venture_name, website)

    # Build the ExtractedFields object from the parsed dict.
    # We use safe_get to handle missing keys gracefully.
    try:
        fields = ExtractedFields(
            venture_name=safe_get(parsed, "venture_name", default=venture_name),
            website=safe_get(parsed, "website", default=website),
            location=safe_get(parsed, "location", default="Unknown"),
            short_description=safe_get(parsed, "short_description", default=""),
            stage=safe_get(parsed, "stage", default="Unknown"),
            total_funding=safe_get(parsed, "total_funding", default="Unknown"),
            insurance_sector=safe_get(parsed, "insurance_sector", default="Unknown"),
        )
        log.debug(
            f"Extractor: '{venture_name}' → "
            f"location={fields.location}, stage={fields.stage}"
        )
        return fields

    except Exception as e:
        log.warning(f"Extractor: failed to build ExtractedFields for '{venture_name}': {e}")
        return _fallback_fields(venture_name, website)


def _fallback_fields(venture_name: str, website: str) -> ExtractedFields:
    """
    Return a minimal ExtractedFields object when extraction fails.

    This ensures the pipeline continues even if a single extraction fails.

    Args:
        venture_name: The company name to include in the fallback.
        website: The confirmed website URL to include in the fallback.

    Returns:
        An ExtractedFields object with all optional fields set to "Unknown".
    """
    return ExtractedFields(
        venture_name=venture_name,
        website=website,
        location="Unknown",
        short_description="",
        stage="Unknown",
        total_funding="Unknown",
        insurance_sector="Unknown",
    )
