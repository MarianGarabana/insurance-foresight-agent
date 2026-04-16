"""
utils.py
--------
Small shared helper functions used across multiple modules.

These are generic utilities — they do not contain domain logic.
Keep domain logic in the relevant module (scout, verifier, etc.).
"""

import json
import re
from typing import Any

from src.logger import get_logger

log = get_logger(__name__)


def parse_json_response(raw_text: str, context: str = "") -> dict | None:
    """
    Safely parse a JSON string returned by the LLM.

    The LLM is instructed to return raw JSON, but sometimes it wraps the
    response in markdown code fences like ```json ... ```. This function
    handles both cases.

    Args:
        raw_text: The raw string content from the LLM response.
        context: Optional label for log messages (e.g. "scout", "classifier").

    Returns:
        A Python dict if parsing succeeded, or None if it failed.
    """
    if not raw_text:
        log.warning(f"[{context}] Received empty response from LLM.")
        return None

    # Strip markdown code fences if present (e.g. ```json ... ```)
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"[{context}] Failed to parse JSON: {e}\nRaw text: {raw_text[:500]}")
        return None


def clean_url(url: str) -> str:
    """
    Normalize a URL for use in HTTP requests.

    Adds https:// if no scheme is present.
    Strips trailing slashes.

    Args:
        url: A raw URL string, e.g. "example.com" or "https://example.com/".

    Returns:
        A normalized URL, e.g. "https://example.com".

    Examples:
        >>> clean_url("example.com")
        'https://example.com'
        >>> clean_url("http://example.com/")
        'http://example.com'
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def safe_get(data: dict, key: str, default: Any = "Unknown") -> Any:
    """
    Get a value from a dict, returning a default if the key is missing or value is None.

    Args:
        data: The dictionary to read from.
        key: The key to look up.
        default: Value to return if key is missing or value is None/empty.

    Returns:
        The value from the dict, or the default.
    """
    value = data.get(key)
    if value is None or value == "":
        return default
    return value


def truncate(text: str, max_length: int = 200) -> str:
    """
    Truncate a string to max_length characters, adding '...' if cut.

    Useful for keeping log messages readable.

    Args:
        text: The string to truncate.
        max_length: Maximum number of characters to keep.

    Returns:
        The original string if short enough, or a truncated version.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
