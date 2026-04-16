"""
classifier.py
-------------
STEP 4 — Classification

The Classifier assigns each verified venture to:
1. One or more driver tags (D1–D11) from our research framework
2. A category: "Direct" (clearly insurtech) or "Indirect" (adjacent)

In Version 1, classification is done by the LLM using a detailed system
prompt that describes all 11 drivers.

The classifier also returns a confidence score (0.0–1.0) and a short
reasoning text, so researchers can audit the classification.

TODO (future improvements):
- Add a rule-based pre-classifier using keyword matching as a first pass.
- Fine-tune a smaller model specifically for this classification task.
- Allow human override / correction of classifications in the output CSV.
- Build a feedback loop: capture human corrections to improve future classifications.
"""

from src.llm_client import call_llm
from src.logger import get_logger
from src.models import ClassificationResult, ExtractedFields, VALID_DRIVERS, VALID_CATEGORIES
from src.prompts import CLASSIFIER_SYSTEM_PROMPT, build_classifier_prompt
from src.utils import parse_json_response, safe_get

log = get_logger(__name__)


def classify_venture(fields: ExtractedFields) -> ClassificationResult:
    """
    Classify a venture into the D1–D11 driver framework.

    Args:
        fields: An ExtractedFields object from the Extractor step.

    Returns:
        A ClassificationResult with driver tags, category, confidence, and reasoning.
        Falls back to a conservative default if the LLM call or parsing fails.
    """
    log.info(f"Classifier: classifying '{fields.venture_name}'")

    # Build the prompt for this specific venture
    user_prompt = build_classifier_prompt(
        venture_name=fields.venture_name,
        website=fields.website,
        short_description=fields.short_description,
        insurance_sector=fields.insurance_sector,
    )

    # Call the LLM via shared client
    raw_response = call_llm(
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context="classifier",
        temperature=0.1,
    )

    # If LLM call failed, return a fallback with low confidence
    if raw_response is None:
        log.warning(f"Classifier: LLM failed for '{fields.venture_name}'. Using fallback.")
        return _fallback_classification(fields.venture_name)

    # Parse the JSON response
    parsed = parse_json_response(raw_response, context="classifier")
    if parsed is None:
        log.warning(f"Classifier: JSON parse failed for '{fields.venture_name}'. Using fallback.")
        return _fallback_classification(fields.venture_name)

    # Extract and validate driver tags
    raw_tags = parsed.get("driver_tags", [])
    validated_tags = _validate_driver_tags(raw_tags, fields.venture_name)

    # Extract and validate category
    raw_category = parsed.get("category", "")
    validated_category = _validate_category(raw_category, fields.venture_name)

    # Extract confidence score (clamp to 0.0–1.0 just in case)
    raw_confidence = parsed.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (ValueError, TypeError):
        confidence = 0.5

    reasoning = safe_get(parsed, "reasoning", default="No reasoning provided.")

    result = ClassificationResult(
        venture_name=fields.venture_name,
        driver_tags=validated_tags,
        category=validated_category,
        confidence=confidence,
        reasoning=reasoning,
    )

    log.debug(
        f"Classifier: '{fields.venture_name}' → {result.driver_tags} / "
        f"{result.category} (confidence={result.confidence:.2f})"
    )
    return result


def _validate_driver_tags(raw_tags: list, venture_name: str) -> list[str]:
    """
    Validate and filter driver tags returned by the LLM.

    Only accepts tags that are in the VALID_DRIVERS list (D1–D11).
    Invalid tags are logged and dropped.

    Args:
        raw_tags: List of tag strings from the LLM (may contain invalid values).
        venture_name: For log messages.

    Returns:
        A list of valid driver tag strings. Falls back to ["D1"] if none are valid.
    """
    if not isinstance(raw_tags, list):
        log.warning(f"Classifier: driver_tags for '{venture_name}' is not a list: {raw_tags}")
        return ["D1"]  # Conservative fallback

    valid_tags = []
    for tag in raw_tags:
        if isinstance(tag, str) and tag.strip().upper() in VALID_DRIVERS:
            valid_tags.append(tag.strip().upper())
        else:
            log.warning(f"Classifier: invalid driver tag '{tag}' for '{venture_name}'. Skipping.")

    if not valid_tags:
        log.warning(f"Classifier: no valid driver tags for '{venture_name}'. Defaulting to D1.")
        return ["D1"]

    return valid_tags


def _validate_category(raw_category: str, venture_name: str) -> str:
    """
    Validate the category returned by the LLM.

    Only "Direct" and "Indirect" are valid. Defaults to "Indirect" if invalid,
    because it is the more conservative assumption.

    Args:
        raw_category: Category string from the LLM.
        venture_name: For log messages.

    Returns:
        "Direct" or "Indirect".
    """
    normalized = raw_category.strip().capitalize() if isinstance(raw_category, str) else ""
    if normalized in VALID_CATEGORIES:
        return normalized

    log.warning(
        f"Classifier: invalid category '{raw_category}' for '{venture_name}'. "
        "Defaulting to 'Indirect'."
    )
    return "Indirect"  # Conservative default


def _fallback_classification(venture_name: str) -> ClassificationResult:
    """
    Return a minimal ClassificationResult when classification fails.

    Keeps the pipeline running. The researcher will see low confidence
    and can manually review this entry.

    Args:
        venture_name: The company name to include in the fallback.

    Returns:
        A ClassificationResult with default/placeholder values.
    """
    return ClassificationResult(
        venture_name=venture_name,
        driver_tags=["D1"],  # D1 is the most common default for this research project
        category="Indirect",
        confidence=0.0,
        reasoning="Classification failed — requires manual review.",
    )


# _call_llm() was removed in v0.2.
# All LLM calls now go through src/llm_client.py → call_llm().
# This eliminated ~40 lines of duplicated code from scout.py, extractor.py,
# and classifier.py.
