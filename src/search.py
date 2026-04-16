"""
search.py
---------
Web search integration for the Scout step.

This module gives the Scout access to real-time web search results,
so it can ground candidate discovery in actual web evidence rather than
relying purely on LLM training knowledge.

Current implementation: Tavily API
Tavily is a search API designed for AI agents. It returns clean, structured
results (title, URL, snippet) without HTML parsing.

Sign up at: https://tavily.com

How this fits into the pipeline:
- scout.py calls search_web() BEFORE asking the LLM for candidates.
- The LLM then acts as a parser/extractor of the search results, not a generator.
- This dramatically reduces hallucination and improves recency.

Graceful degradation:
- If TAVILY_API_KEY is not set, search_web() returns an empty list and logs a warning.
- The Scout will then fall back to LLM-only mode (v0.1 behavior).
- This means the system still works without a search key — it's just less grounded.

TODO (future improvements):
- Add SerpAPI as an alternative search backend.
- Add Perplexity API as another option.
- Add a configurable "search_provider" setting to switch between backends.
- Cache search results to avoid redundant API calls for similar queries.
"""

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# Import Tavily at module level so it can be patched in tests.
# We catch ImportError so the rest of the system still imports fine
# even if tavily-python is not installed.
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore[assignment,misc]


# Type alias for clarity: a search result is a dict with title, url, and snippet.
# We use a plain dict (not a Pydantic model) to keep this module simple.
SearchResult = dict  # {"title": str, "url": str, "snippet": str}


def search_web(query: str, max_results: int = 10) -> list[SearchResult]:
    """
    Search the web for a query and return structured results.

    Each result contains:
    - title: the page title
    - url: the page URL
    - snippet: a short excerpt from the page

    If no search API is configured, returns an empty list with a warning.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A list of SearchResult dicts, or an empty list if search is unavailable.
    """
    if not settings.tavily_api_key:
        log.warning(
            "search.py: No TAVILY_API_KEY configured. "
            "Skipping web search — Scout will use LLM-only mode. "
            "Add TAVILY_API_KEY to your .env file to enable search-grounded discovery."
        )
        return []

    return _search_tavily(query=query, max_results=max_results)


def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    """
    Call the Tavily Search API and return normalized results.

    Tavily is designed for AI agents and returns clean text snippets
    without requiring HTML parsing.

    Args:
        query: Search query.
        max_results: Max number of results to request.

    Returns:
        A list of SearchResult dicts, or an empty list on error.
    """
    if TavilyClient is None:
        log.error(
            "tavily-python is not installed. "
            "Run: pip install tavily-python"
        )
        return []

    log.info(f"search.py: Searching Tavily for: '{query}' (max {max_results} results)")

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # "basic" is faster and cheaper than "advanced"
        )
    except Exception as e:
        log.error(f"search.py: Tavily API error: {e}")
        return []

    # Normalize Tavily results into our simple SearchResult format.
    # Tavily returns: {"results": [{"title", "url", "content", "score", ...}]}
    raw_results = response.get("results", [])
    normalized: list[SearchResult] = []

    for item in raw_results:
        url = item.get("url", "")
        title = item.get("title", "")
        # Tavily calls the text "content" — we expose it as "snippet"
        snippet = item.get("content", "")

        if not url:
            continue  # Skip results with no URL

        normalized.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    log.info(f"search.py: Got {len(normalized)} search results.")
    log.debug(f"search.py: Result URLs: {[r['url'] for r in normalized]}")
    return normalized


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """
    Format search results into a readable block for inclusion in an LLM prompt.

    The LLM will parse this text to extract company names.

    Args:
        results: List of SearchResult dicts from search_web().

    Returns:
        A formatted string to embed in the Scout prompt.

    Example output:
        1. Title: FloodSense - Flood Risk Analytics
           URL: https://floodsense.io
           Snippet: FloodSense helps insurers quantify flood exposure...

        2. Title: ...
    """
    if not results:
        return "(no search results available)"

    lines = []
    for i, result in enumerate(results, start=1):
        lines.append(f"{i}. Title: {result.get('title', '')}")
        lines.append(f"   URL: {result.get('url', '')}")
        snippet = result.get("snippet", "")
        # Truncate very long snippets so the prompt doesn't become too long
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(f"   Snippet: {snippet}")
        lines.append("")  # Blank line between results

    return "\n".join(lines)
