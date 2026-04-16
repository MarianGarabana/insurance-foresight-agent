"""
tests/test_search.py
--------------------
Unit tests for src/search.py.

These tests do NOT make real HTTP or API calls.
We mock the Tavily client to test the logic in isolation.

Run with:
    pytest tests/test_search.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.search import search_web, format_results_for_prompt, _search_tavily


class TestFormatResultsForPrompt:
    """Tests for the prompt formatting helper — no mocking needed."""

    def test_empty_results_returns_placeholder(self):
        """Empty list should return a clear placeholder string."""
        result = format_results_for_prompt([])
        assert "(no search results available)" in result

    def test_formats_single_result(self):
        """A single result should appear in the output."""
        results = [
            {"title": "FloodSense - Flood Risk", "url": "https://floodsense.io", "snippet": "AI for flood risk."}
        ]
        output = format_results_for_prompt(results)
        assert "FloodSense" in output
        assert "https://floodsense.io" in output
        assert "AI for flood risk." in output

    def test_formats_multiple_results(self):
        """Multiple results should all appear in the output."""
        results = [
            {"title": "Company A", "url": "https://a.com", "snippet": "Description A."},
            {"title": "Company B", "url": "https://b.com", "snippet": "Description B."},
        ]
        output = format_results_for_prompt(results)
        assert "Company A" in output
        assert "Company B" in output
        assert "https://a.com" in output
        assert "https://b.com" in output

    def test_truncates_long_snippets(self):
        """Snippets longer than 300 characters should be truncated with '...'"""
        long_snippet = "x" * 500
        results = [{"title": "Co", "url": "https://co.com", "snippet": long_snippet}]
        output = format_results_for_prompt(results)
        assert "..." in output

    def test_short_snippet_not_truncated(self):
        """Short snippets should not be truncated."""
        results = [{"title": "Co", "url": "https://co.com", "snippet": "Short snippet."}]
        output = format_results_for_prompt(results)
        assert "Short snippet." in output
        assert "..." not in output

    def test_numbered_results(self):
        """Results should be numbered starting from 1."""
        results = [
            {"title": "A", "url": "https://a.com", "snippet": ""},
            {"title": "B", "url": "https://b.com", "snippet": ""},
        ]
        output = format_results_for_prompt(results)
        assert "1." in output
        assert "2." in output


class TestSearchWebNoApiKey:
    """Tests for search_web() when no API key is configured."""

    def test_returns_empty_list_when_no_api_key(self):
        """search_web() should return [] and not crash when TAVILY_API_KEY is not set."""
        with patch("src.search.settings") as mock_settings:
            mock_settings.tavily_api_key = ""  # No key configured
            result = search_web("find flood ventures")
        assert result == []

    def test_logs_warning_when_no_api_key(self, caplog):
        """search_web() should log a warning when no API key is set."""
        import logging
        with patch("src.search.settings") as mock_settings:
            mock_settings.tavily_api_key = ""
            with caplog.at_level(logging.WARNING):
                search_web("find flood ventures")
        assert any("TAVILY_API_KEY" in record.message for record in caplog.records)


class TestSearchTavilyMocked:
    """Tests for _search_tavily() with a mocked Tavily client."""

    def _make_tavily_response(self, results: list[dict]) -> dict:
        """Helper to build a fake Tavily API response."""
        return {"results": results}

    def test_returns_normalized_results(self):
        """Should normalize Tavily response into our SearchResult format."""
        fake_response = self._make_tavily_response([
            {
                "title": "FloodSense raises $12M",
                "url": "https://techcrunch.com/floodsense",
                "content": "FloodSense is a startup...",
                "score": 0.9,
            }
        ])

        mock_client = MagicMock()
        mock_client.search.return_value = fake_response

        with patch("src.search.TavilyClient", return_value=mock_client):
            with patch("src.search.settings") as mock_settings:
                mock_settings.tavily_api_key = "tvly-fake-key"
                results = _search_tavily(query="flood ventures", max_results=5)

        assert len(results) == 1
        assert results[0]["title"] == "FloodSense raises $12M"
        assert results[0]["url"] == "https://techcrunch.com/floodsense"
        assert results[0]["snippet"] == "FloodSense is a startup..."

    def test_skips_results_with_no_url(self):
        """Results without a URL should be skipped."""
        fake_response = self._make_tavily_response([
            {"title": "No URL result", "url": "", "content": "Some text."},
            {"title": "Valid result", "url": "https://valid.com", "content": "Text."},
        ])

        mock_client = MagicMock()
        mock_client.search.return_value = fake_response

        with patch("src.search.TavilyClient", return_value=mock_client):
            with patch("src.search.settings") as mock_settings:
                mock_settings.tavily_api_key = "tvly-fake-key"
                results = _search_tavily(query="test", max_results=5)

        assert len(results) == 1
        assert results[0]["url"] == "https://valid.com"

    def test_returns_empty_on_api_error(self):
        """If Tavily raises an exception, return empty list without crashing."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API timeout")

        with patch("src.search.TavilyClient", return_value=mock_client):
            with patch("src.search.settings") as mock_settings:
                mock_settings.tavily_api_key = "tvly-fake-key"
                results = _search_tavily(query="test", max_results=5)

        assert results == []

    def test_returns_empty_results_list(self):
        """If Tavily returns empty results, return empty list."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("src.search.TavilyClient", return_value=mock_client):
            with patch("src.search.settings") as mock_settings:
                mock_settings.tavily_api_key = "tvly-fake-key"
                results = _search_tavily(query="test", max_results=5)

        assert results == []
