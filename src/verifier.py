"""
verifier.py
-----------
STEP 2 — Venture Verification

The Verifier takes a candidate venture and tries to confirm it actually exists
and has an official website.

HARD RULE: If no official website can be confirmed, the venture is REJECTED.

v0.2 — Three-layer verification
---------------------------------
Each candidate now goes through three checks in sequence:

  1. Domain sanity check
     Reject URLs that point to directories, aggregators, or news sites
     (LinkedIn, Crunchbase, Wikipedia, TechCrunch, etc.).
     These are not the company's own website.

  2. HTTP reachability check
     Try to reach the URL. Reject if it returns an error or times out.

  3. Content matching check
     Fetch the page text and check if the company name appears in it.
     This catches cases where we reached a URL but it's the wrong company
     (e.g. the domain was taken by someone else, or the LLM guessed a wrong URL).

A candidate is only verified if it passes ALL three checks.

Rejection reasons are explicit and specific, so researchers know exactly
why each candidate was dropped.

TODO (future improvements):
- Use a headless browser (Playwright) for JS-rendered pages.
- Add LLM-based content verification for ambiguous cases.
- Add retry logic with exponential backoff.
"""

from urllib.parse import urlparse

import requests

from src.config import settings
from src.logger import get_logger
from src.models import CandidateVenture, VerificationResult
from src.utils import clean_url

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# HTTP status codes we consider "success" (site is reachable and not broken)
_ACCEPTABLE_STATUS_CODES = set(range(200, 400))  # 200–399

# Domains we reject immediately — these are directories, aggregators, or news
# sites, not official company websites.
# If the Scout suggested one of these, it means the LLM confused a news article
# URL with the company's own domain.
_DIRECTORY_DOMAINS: set[str] = {
    "linkedin.com",
    "www.linkedin.com",
    "crunchbase.com",
    "www.crunchbase.com",
    "pitchbook.com",
    "www.pitchbook.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "techcrunch.com",
    "www.techcrunch.com",
    "businesswire.com",
    "www.businesswire.com",
    "prnewswire.com",
    "www.prnewswire.com",
    "globenewswire.com",
    "www.globenewswire.com",
    "reuters.com",
    "www.reuters.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "forbes.com",
    "www.forbes.com",
    "angellist.com",
    "www.angellist.com",
    "wellfound.com",
    "www.wellfound.com",
    "f6s.com",
    "www.f6s.com",
    "owler.com",
    "www.owler.com",
    "zoominfo.com",
    "www.zoominfo.com",
    "dnb.com",
    "www.dnb.com",
    "opencorporates.com",
    "www.opencorporates.com",
}

# Common headers to make our request look like a real browser.
# Some websites block requests without a User-Agent header.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; InsuranceForesightBot/1.0; "
        "+https://github.com/your-org/insurance-foresight-agent)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# How many characters of the page to fetch for content matching.
# We don't need the whole page — the company name should appear early.
_CONTENT_CHECK_BYTES = 3000

# Stop words to ignore when matching company name against page content.
# These words appear everywhere and don't help confirm identity.
_STOP_WORDS = {
    "inc", "inc.", "corp", "corp.", "llc", "ltd", "limited", "group",
    "company", "co", "co.", "the", "and", "of", "for", "by",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def verify_venture(candidate: CandidateVenture) -> VerificationResult:
    """
    Attempt to verify that a candidate venture exists and has a live website.

    Runs three checks in order:
    1. Domain sanity (not a directory/aggregator)
    2. HTTP reachability (site is live)
    3. Content matching (page mentions the company name)

    Args:
        candidate: A CandidateVenture produced by the Scout.

    Returns:
        VerificationResult with verified=True and a website URL if all checks pass,
        or verified=False with a specific rejection_reason if any check fails.
    """
    log.info(f"Verifier: verifying '{candidate.name}'")

    # No website at all — cannot verify.
    if not candidate.suggested_website:
        return _reject(
            candidate,
            reason=(
                "No official website was provided by the Scout. "
                "This usually means the search results did not mention the company's domain."
            ),
        )

    # Build URL variants to try (with and without www.)
    urls_to_try = _build_url_variants(candidate.suggested_website)

    # ── Check 1: Domain sanity ──────────────────────────────────────────── #
    for url in urls_to_try:
        domain = _extract_domain(url)
        if domain in _DIRECTORY_DOMAINS:
            log.warning(
                f"Verifier: '{candidate.name}' → domain '{domain}' is a directory/aggregator. Rejecting."
            )
            return _reject(
                candidate,
                reason=(
                    f"The suggested URL '{url}' points to a directory or aggregator site "
                    f"({domain}), not the company's own official website."
                ),
            )

    # ── Checks 2 + 3: HTTP + content ───────────────────────────────────── #
    for url in urls_to_try:
        http_result = _try_url(url)

        if http_result is None:
            log.debug(f"Verifier: '{url}' not reachable.")
            continue  # Try next URL variant

        status_code, final_url, page_text = http_result

        # Check 3: Does the page text mention the company name?
        if not is_content_matching_company(name=candidate.name, text=page_text):
            log.warning(
                f"Verifier: '{candidate.name}' — page at {final_url} "
                "does not appear to be the company's official site. Rejecting."
            )
            return _reject(
                candidate,
                reason=(
                    f"Reached '{final_url}' (HTTP {status_code}) but the page content "
                    f"does not appear to describe the company '{candidate.name}'. "
                    "This may be the wrong domain or a redirected page."
                ),
                http_status=status_code,
            )

        # All checks passed!
        log.info(f"Verifier: '{candidate.name}' verified at {final_url} (HTTP {status_code})")
        return VerificationResult(
            candidate_name=candidate.name,
            verified=True,
            website=final_url,
            http_status=status_code,
        )

    # All URL variants failed HTTP
    log.warning(f"Verifier: could not reach any URL for '{candidate.name}'. Rejecting.")
    return _reject(
        candidate,
        reason=(
            f"Could not reach any URL variant of '{candidate.suggested_website}'. "
            "The site may be down, blocking automated requests, or the URL may be wrong."
        ),
    )


def is_content_matching_company(name: str, text: str) -> bool:
    """
    Check whether a web page's text appears to describe a specific company.

    Simple approach: check if the significant words from the company name
    appear in the page text (case-insensitive). This catches the most common
    failure mode where we reached the right domain but got a generic or
    wrong page.

    We ignore common stop words (Inc, Ltd, Group, etc.) since these appear
    everywhere and don't help distinguish companies.

    Args:
        name: The company name to look for (e.g. "FloodSense Analytics").
        text: The raw page text to search in.

    Returns:
        True if the company appears to be described on this page, False otherwise.

    Examples:
        >>> is_content_matching_company("FloodSense", "FloodSense helps insurers...")
        True
        >>> is_content_matching_company("FloodSense", "Page not found. 404 error.")
        False
    """
    if not text:
        return False
    if not name:
        # No name to check against — let the venture through.
        # The pipeline has other quality gates (HTTP, domain check).
        return True

    text_lower = text.lower()

    # Split the company name into meaningful words, ignoring stop words
    name_words = [
        word.strip(".,()-").lower()
        for word in name.split()
        if word.strip(".,()-").lower() not in _STOP_WORDS
        and len(word.strip(".,()-")) > 2  # Skip very short words
    ]

    if not name_words:
        # Company name is all stop words (very unlikely, but handle gracefully)
        return True

    # Check how many of the meaningful name words appear in the page
    matching_words = [word for word in name_words if word in text_lower]
    match_ratio = len(matching_words) / len(name_words)

    # Require at least 50% of meaningful words to appear.
    # This is intentionally lenient to handle cases like "FloodSense" appearing
    # as "Flood Sense" or abbreviations on the page.
    is_match = match_ratio >= 0.5

    log.debug(
        f"Content match for '{name}': {len(matching_words)}/{len(name_words)} words found "
        f"({int(match_ratio * 100)}%) → {'PASS' if is_match else 'FAIL'}"
    )
    return is_match


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reject(
    candidate: CandidateVenture,
    reason: str,
    http_status: int | None = None,
) -> VerificationResult:
    """
    Build a VerificationResult with verified=False.

    Helper to avoid repeating the same construction in multiple places.

    Args:
        candidate: The candidate being rejected.
        reason: Specific explanation for the rejection.
        http_status: HTTP status code if we reached the server (optional).

    Returns:
        A VerificationResult with verified=False.
    """
    return VerificationResult(
        candidate_name=candidate.name,
        verified=False,
        website=candidate.suggested_website,
        http_status=http_status,
        rejection_reason=reason,
    )


def _build_url_variants(suggested_url: str) -> list[str]:
    """
    Build a small list of URL variants to try for a given suggested URL.

    Tries the URL as-is and also with/without www. prefix.

    Args:
        suggested_url: The URL suggested by the Scout.

    Returns:
        A list of normalized URL strings to try in order.
    """
    cleaned = clean_url(suggested_url)
    variants = [cleaned]

    parsed = urlparse(cleaned)
    host = parsed.netloc

    if host and not host.startswith("www."):
        www_variant = cleaned.replace(host, f"www.{host}", 1)
        variants.append(www_variant)

    return variants


def _extract_domain(url: str) -> str:
    """
    Extract the hostname (domain) from a URL.

    Args:
        url: A full URL string.

    Returns:
        The domain as a lowercase string, e.g. "linkedin.com".
        Returns empty string if parsing fails.
    """
    try:
        return urlparse(clean_url(url)).netloc.lower()
    except Exception:
        return ""


def _try_url(url: str) -> tuple[int, str, str] | None:
    """
    Try to make an HTTP GET request to a URL.

    Returns status code, final URL (after redirects), and a snippet of page text.
    The page text is used for content matching in check 3.

    Args:
        url: The full URL to try.

    Returns:
        A tuple of (http_status_code, final_url, page_text_snippet) if successful,
        or None if the request failed.
    """
    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            timeout=settings.request_timeout,
            allow_redirects=True,
            stream=True,  # Stream so we can read only the first N bytes
        )
        log.debug(f"Verifier: {url} → HTTP {response.status_code}")

        if response.status_code not in _ACCEPTABLE_STATUS_CODES:
            return None

        # Read only the first _CONTENT_CHECK_BYTES to keep this fast.
        # We decode with errors="replace" to handle encoding issues gracefully.
        raw_bytes = response.raw.read(_CONTENT_CHECK_BYTES)
        page_text = raw_bytes.decode("utf-8", errors="replace")

        return (response.status_code, response.url, page_text)

    except requests.exceptions.SSLError:
        log.debug(f"Verifier: SSL error for {url}, trying without SSL verification.")
        try:
            response = requests.get(
                url,
                headers=_HEADERS,
                timeout=settings.request_timeout,
                allow_redirects=True,
                verify=False,  # noqa: S501 — intentional fallback for self-signed certs
                stream=True,
            )
            if response.status_code in _ACCEPTABLE_STATUS_CODES:
                raw_bytes = response.raw.read(_CONTENT_CHECK_BYTES)
                page_text = raw_bytes.decode("utf-8", errors="replace")
                return (response.status_code, response.url, page_text)
        except Exception:
            pass
        return None

    except requests.exceptions.ConnectionError:
        log.debug(f"Verifier: connection error for {url}")
        return None
    except requests.exceptions.Timeout:
        log.debug(f"Verifier: timeout for {url} after {settings.request_timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        log.debug(f"Verifier: request error for {url}: {e}")
        return None
